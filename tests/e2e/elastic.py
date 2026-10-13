# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Two kvcached servers on one GPU: memory one releases, the other can use.

Each server's KV usage is read from its kvcached segments in /dev/shm, and the
load is sized from a calibration burst, so the checks do not depend on how
fast requests finish:
  1. A gets enough concurrent long requests to need most of the free memory;
  2. A finishes, its prefix cache is reset, and it must give at least half
     of that memory back;
  3. B gets its own such load, and must grow past what was free while A was
     at its peak, which it can only do with the memory A gave back;
  4. both get that load at once, more than fits: requests may queue, but all
     must finish and neither server may crash.
Greedy probes before and after must give the same tokens on each server.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import shlex
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from e2e.memory import MemoryProbe
from e2e import client as cl
from e2e.matrix import BATCH_FLAG, ELASTIC_BATCH, Case, ElasticPair
from e2e.run import (
    CaseResult,
    Container,
    Server,
    case_setup,
    check_gpu_released,
    launch,
    stop_and_check,
)

# Distinct prompts, so the prefix cache shares nothing between requests, and
# ignore_eos, so every request holds its KV until it reaches MAX_TOKENS.
PROMPT_WORDS = 3000
MAX_TOKENS = 1024
# Each engine's load is sized to need this fraction of the free memory.
TARGET = 0.7
MIB = 1 << 20
SEGMENT = struct.Struct("=3q")
# A snapshot of two individually locked segments must not straddle our transfer
# commands, or it could combine the donor's old limit with the recipient's new one.
_SEGMENT_SNAPSHOT_LOCK = threading.RLock()

# For setup.json (the CI results issue shows it); keep in step with run_elastic.
WORKLOAD = (
    f"both servers use the contiguous KV layout and a batch limit of {ELASTIC_BATCH}; "
    "A starts first, then B, on the same GPU",
    f"every load request is a distinct {PROMPT_WORDS}-word filler prompt with "
    f"{MAX_TOKENS} output tokens and ignore_eos",
    "calibration: 4 such requests per server measure the KV memory one request holds",
    f"each server's load is sized to need {TARGET:.0%} of the GPU memory free after "
    f"calibration, at most {ELASTIC_BATCH} requests",
    "phases: (1) A gets its load; (2) A's prefix cache is reset; (3) B gets its load and "
    "its prefix cache is reset; (4) both get their loads at the same time",
    "with a bounded KV budget: calibrate with half each, transfer 85% to A, then "
    "85% to B after A shrinks, and restore half each for the simultaneous load; "
    "reserve 1–2 page bundles per server and verify used plus reserved bytes",
    "3 greedy probe prompts of 32 tokens on each server before and after",
)
CHECKS = {
    "<server>_ready": "the server answers /health within the ready timeout",
    "a_requests_ok, b_requests_ok": "every request of the server's own load succeeds",
    "a_filled_memory": "A's KV usage grows by more than half of the free memory",
    "a_released_memory": "after its prefix cache is reset, A keeps less than half of "
                         "what it grew",
    "b_reused_a_memory": "B's KV usage grows past what was free at A's peak",
    "b_reused_budget": "B grows beyond its allowance before A's budget was transferred",
    "shared_budget_enforced": "both requested limits and mapped KV bytes stay within "
                              "one shared budget, with shrink completed before expansion",
    "contended_requests_ok": "every request succeeds while both loads run",
    "<server>_probe_unchanged": "the probes give the same tokens before and after",
    "<server>_alive": "the server is still running at the end",
    "<server>_<shutdown check>": "the shutdown and cleanup checks of each server",
}


def read_segments(directory: str = "/dev/shm") -> dict[str, dict[str, int]]:
    """All KV counters of every kvcached segment, read as kvctl does:
    three int64s (limit, used, preallocated) under a shared flock. /dev/shm
    is shared with the containers (--ipc=host), and this is much faster than
    running kvctl once a second."""
    with _SEGMENT_SNAPSHOT_LOCK:
        return _read_segments(directory)


def _read_segments(directory: str) -> dict[str, dict[str, int]]:
    segments = {}
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        try:
            if os.path.getsize(path) != SEGMENT.size:
                continue
            with open(path, "rb") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                limit, in_use, reserved = SEGMENT.unpack(f.read(SEGMENT.size))
        except (OSError, struct.error):
            continue
        # A requested zero limit must not hide pages still awaiting release.
        if min(limit, in_use, reserved) >= 0:
            segments[name] = {"limit": limit, "used": in_use, "preallocated": reserved}
    return segments


def server_segments(srv: Server, segments: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {name: state for name, state in segments.items()
            if name == srv.ipc or (name.startswith(srv.ipc + "_g")
                                  and name[len(srv.ipc) + 2:].isdigit())}


class SharedBudget:
    """Transfer one physical KV budget, waiting for shrink before any expansion.

    The bounded release model has one KV pool per instance. Refuse other pool
    layouts rather than silently limiting only one group of a hybrid model.
    """

    def __init__(self, servers: list[tuple[Container, Server]], total: int,
                 timeout: float = 60):
        self.servers = servers
        self.total = total
        self.timeout = timeout
        self.history: list[dict] = []
        self.established = False

    @staticmethod
    def state(srv: Server, segments: Optional[dict] = None) -> dict[str, int]:
        states = server_segments(srv, read_segments() if segments is None else segments)
        if set(states) != {srv.ipc}:
            raise RuntimeError(f"bounded KV test requires one live segment for {srv.ipc}")
        return states[srv.ipc]

    def _set(self, host: Container, srv: Server, target: int, previous: int) -> None:
        with _SEGMENT_SNAPSHOT_LOCK:
            proc = host.exec(f"kvctl limit {shlex.quote(srv.ipc)} {target}")
        if proc.returncode or b"Error:" in proc.stderr:
            raise RuntimeError(f"setting KV budget failed: {proc.stderr!r}")
        # Some integrations observe a new limit at their next scheduler step.
        # A tiny request drives that step; reset it before measuring the drain.
        if target < previous and target:
            if not srv.client.generate(cl.BASE_PROMPTS[0], 1).ok:
                raise RuntimeError(f"budget shrink probe failed for {srv.ipc}")
            if not srv.client.reset_prefix_cache():
                raise RuntimeError(f"budget shrink cache reset failed for {srv.ipc}")
        deadline = time.monotonic() + self.timeout
        while True:
            state = self.state(srv)
            if state["limit"] == target and state["used"] + state["preallocated"] <= target:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"KV budget did not settle for {srv.ipc}: {state}, target={target}")
            time.sleep(0.25)

    def within_budget(self, segments: dict) -> bool:
        states = [self.state(srv, segments) for _, srv in self.servers]
        return (sum(s["limit"] for s in states) <= self.total
                and sum(s["used"] + s["preallocated"] for s in states) <= self.total)

    def set_shares(self, shares: tuple[int, int], stage: str) -> None:
        if len(shares) != len(self.servers) or min(shares) < 0 or sum(shares) > self.total:
            raise ValueError("instance allowances must fit the shared KV budget")
        old = [self.state(srv)["limit"] for _, srv in self.servers]
        # Never grant the recipient capacity until the donor has actually unmapped
        # enough pages, including its otherwise invisible preallocation reserve.
        for shrinking in (True, False):
            for (host, srv), previous, target in zip(self.servers, old, shares):
                if target != previous and (target < previous) == shrinking:
                    self._set(host, srv, target, previous)
                    if self.established and not self.within_budget(read_segments()):
                        raise RuntimeError("shared KV budget exceeded during transfer")
        segments = read_segments()
        if not self.within_budget(segments):
            raise RuntimeError("shared KV budget was not established")
        self.established = True
        self.history.append({"stage": stage, "t": time.time(), "segments": {
            srv.ipc: dict(self.state(srv, segments)) for _, srv in self.servers}})


class Usage:
    """Sample the kvcached segments and the GPU's memory once a second."""

    def __init__(self, probe: MemoryProbe) -> None:
        self.probe = probe
        self.error = None
        self.samples: list[tuple[float, int, dict[str, dict[str, int]]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def of(srv: Server, segments: dict[str, dict[str, int]], field: str = "used") -> int:
        # A server can have one segment per KV cache group: <ipc>, <ipc>_g1, ...
        return sum(v[field] for v in server_segments(srv, segments).values())

    def used(self, srv: Server) -> int:
        return self.of(srv, read_segments())

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append((time.time(), self.probe.read()[0], read_segments()))
            except Exception as e:
                self.error = e
                return
            self._stop.wait(0.25)

    def peak(self, srv: Server, since: float) -> int:
        if self.error:
            raise RuntimeError("GPU sampling failed") from self.error
        return max((self.of(srv, u) for t, _, u in self.samples if t >= since), default=0)

    def gpu_peak_mib(self, since: float) -> int:
        return max((g for t, g, _ in self.samples if t >= since), default=0)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def burst(srv: Server, pool: ThreadPoolExecutor, n: int, seed: int) -> list[cl.Result]:
    """Send n distinct long requests at once and wait for all of them."""
    futures = [pool.submit(srv.client.generate, cl.long_prompt(PROMPT_WORDS, seed + i),
                           MAX_TOKENS, True) for i in range(n)]
    return [f.result() for f in futures]


def drain(usage: Usage, srv: Server, timeout: float = 60) -> int:
    """Reset the server's prefix cache, so that finished requests hold no KV,
    and wait until its usage stops falling; return it."""
    if not srv.client.reset_prefix_cache():
        raise RuntimeError("prefix-cache reset failed")
    last = usage.used(srv)
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(5)
        now = usage.used(srv)
        if now == 0 or (now >= last and time.time() >= deadline - timeout + 15):
            return now
        last = now
    return last


def start_ballast(ct: Container, mib: int) -> str:
    """Hold `mib` of GPU memory, to emulate a smaller GPU when testing locally."""
    (ct.out_dir / "elastic-ballast").mkdir(parents=True, exist_ok=True)
    code = ("import time, torch; "
            f"x = torch.empty({mib} << 20, dtype=torch.uint8, device='cuda'); "
            "print('ballast ready', flush=True); time.sleep(1e9)")
    pid = ct.start_server("elastic-ballast", {}, ["python3", "-c", code])
    deadline = time.time() + 120
    while time.time() < deadline:
        log = ct.out_dir / "elastic-ballast" / "serve.log"
        if log.exists() and "ballast ready" in log.read_text(errors="replace"):
            return pid
        time.sleep(1)
    raise RuntimeError("GPU ballast did not start")


def run_elastic(ct_a: Container, ct_b: Container, pair: ElasticPair, run_id: str,
                base_port: int, timeout: float, emulate_gpu_mib: int = 0, budget_mib: int = 0) -> CaseResult:
    """Server A runs in ct_a and server B in ct_b; both use GPU ct_a.gpu."""
    res = CaseResult(pair.name)
    gpu = ct_a.gpu
    try:
        probe = MemoryProbe(gpu)
    except Exception as e:
        res.add("memory_probe", False, str(e))
        return res
    res.log["memory_source"] = probe.source
    res.log["budget_mib"] = budget_mib
    ballast: Optional[str] = None
    case_a = Case(pair.engine_a, pair.model_a, "c1", tag="elastic-a")
    case_b = Case(pair.engine_b, pair.model_b, "c1", tag="elastic-b")
    setup = {"kind": "elastic", "gpu": gpu, "batch": ELASTIC_BATCH,
             "a": case_setup(case_a), "b": case_setup(case_b)}
    res.log["setup"] = setup
    started: list[tuple[Container, Server]] = []
    usage: Optional[Usage] = None
    budget: Optional[SharedBudget] = None
    budget_start = 0.0
    try:
        if emulate_gpu_mib:
            _, total = probe.read()
            if total > emulate_gpu_mib:
                ballast = start_ballast(ct_a, total - emulate_gpu_mib)
                res.log["ballast_mib"] = total - emulate_gpu_mib
        # Start one after the other so the second sees the first's weights.
        for host, case, port, extra in ((ct_a, case_a, base_port + 1, pair.extra_args_a),
                                        (ct_b, case_b, base_port + 2, pair.extra_args_b)):
            batch = (BATCH_FLAG[case.engine], str(ELASTIC_BATCH))
            extra_env = ({"KVCACHED_MIN_RESERVED_PAGES": "1",
                          "KVCACHED_MAX_RESERVED_PAGES": "2"} if budget_mib else None)
            srv = launch(host, case, run_id, port, extra_args=(*batch, *extra),
                         extra_env=extra_env)
            setup["a" if case is case_a else "b"] = srv.setup
            started.append((host, srv))
            ready = srv.client.wait_ready(timeout, lambda: host.alive(srv.pid))
            res.add(f"{case.name}_ready", ready)
            if not ready:
                return res
        (_, a), (_, b) = started
        probes = cl.BASE_PROMPTS[:3]
        base = {s.ipc: s.client.run_phase("probe", probes, 32, 1) for _, s in started}
        if budget_mib:
            budget = SharedBudget(started, budget_mib * MIB)
            half = budget.total // 2
            budget.set_shares((half, budget.total - half), "calibration")
            budget_start = time.time()
            res.log["budget_transfers"] = budget.history
            res.log["reuse_scope"] = "one shared KV budget with shrink-before-expand transfer"
        usage = Usage(probe)
        mem: dict[str, float] = {}
        res.log["memory_mib"] = mem
        res.log["segments"] = {k: v for k, v in read_segments().items()
                               if k.startswith((a.ipc, b.ipc))}

        with ThreadPoolExecutor(max_workers=4 * ELASTIC_BATCH) as pool:
            # Calibrate: the KV memory one request of this shape holds.
            per_request = {}
            for srv in (a, b):
                before = usage.used(srv)
                t = time.time()
                calibration = burst(srv, pool, 4, 1_000)
                if not all(r.ok for r in calibration):
                    raise RuntimeError("elastic calibration request failed")
                per_request[srv.ipc] = max(usage.peak(srv, t) - before, MIB) / 4
                drain(usage, srv)
            idle_gpu, total = probe.read()
            free = (total - idle_gpu) * MIB
            if budget_mib:
                if free < budget_mib * MIB:
                    raise RuntimeError("insufficient free memory for the bounded KV test")
                free = budget_mib * MIB
            # More than the batch limit would only queue.
            n = {s.ipc: min(math.ceil(TARGET * free / per_request[s.ipc]), ELASTIC_BATCH)
                 for s in (a, b)}
            mem.update(free=free / MIB, per_request_a=per_request[a.ipc] / MIB,
                       per_request_b=per_request[b.ipc] / MIB, n_a=n[a.ipc], n_b=n[b.ipc])

            # 1. A takes most of the free memory.
            if budget is not None:
                small = budget.total * 15 // 100
                large = budget.total - small
                budget.set_shares((large, small), "a-heavy")
                previous_b_allowance = small
            a_idle = usage.used(a)
            t = time.time()
            results = burst(a, pool, n[a.ipc], 10_000)
            a_growth = usage.peak(a, t) - a_idle
            free_at_a_peak = (total - usage.gpu_peak_mib(t)) * MIB
            physical_free_at_a_peak = free_at_a_peak
            if budget_mib:
                free_at_a_peak = free - a_growth
            res.add("a_requests_ok", all(r.ok for r in results),
                    f"{sum(not r.ok for r in results)}/{len(results)} failed")
            res.add("a_filled_memory", a_growth > 0.5 * free,
                    f"A grew {a_growth / MIB:.0f} MiB of {free / MIB:.0f} MiB free")

            # 2. A gives its KV memory back.
            a_after = drain(usage, a)
            if budget is not None:
                budget.set_shares((small, large), "b-heavy")
                a_after = usage.used(a)
            mem["gpu_free_after_a"] = total - probe.read()[0]
            if budget_mib:
                released = mem["gpu_free_after_a"] * MIB - physical_free_at_a_peak
                mem["physical_release_after_a"] = released / MIB
                # Require driver-visible release too, not only a counter change.
                # 64 MiB tolerates small allocator/context accounting fluctuations.
                res.add("a_physical_memory_released", a_growth > 0 and released + 64 * MIB >= 0.5 * a_growth,
                        f"driver reports {released / MIB:.0f} MiB released after {a_growth / MIB:.0f} MiB KV growth")
            res.add("a_released_memory", a_after - a_idle < 0.5 * a_growth,
                    f"A still holds {(a_after - a_idle) / MIB:.0f} of the "
                    f"{a_growth / MIB:.0f} MiB it grew")

            # 3. B grows past what was free while A was at its peak.
            b_idle = usage.used(b)
            t = time.time()
            results = burst(b, pool, n[b.ipc], 20_000)
            b_growth = usage.peak(b, t) - b_idle
            res.add("b_requests_ok", all(r.ok for r in results),
                    f"{sum(not r.ok for r in results)}/{len(results)} failed")
            reuse_threshold = max(free_at_a_peak, previous_b_allowance) if budget else free_at_a_peak
            res.add("b_reused_budget" if budget_mib else "b_reused_a_memory", b_growth > reuse_threshold,
                    f"B grew {b_growth / MIB:.0f} MiB; {free_at_a_peak / MIB:.0f} MiB "
                    "was free at A's peak" + (f"; B previously had {previous_b_allowance / MIB:.0f} MiB"
                                               if budget else ""))
            drain(usage, b)

            # 4. Both at once.
            if budget is not None:
                budget.set_shares((half, budget.total - half), "simultaneous")
            fa = pool.submit(burst, a, pool, n[a.ipc], 30_000)
            fb = pool.submit(burst, b, pool, n[b.ipc], 40_000)
            results = fa.result() + fb.result()
            res.add("contended_requests_ok", all(r.ok for r in results),
                    f"{sum(not r.ok for r in results)}/{len(results)} failed")
            mem.update(a_growth=a_growth / MIB, a_after=(a_after - a_idle) / MIB,
                       free_at_a_peak=free_at_a_peak / MIB, b_growth=b_growth / MIB)

        for host, srv in started:
            after = srv.client.run_phase("probe", probes, 32, 1)
            same = base[srv.ipc].all_ok and all(x.token_ids == y.token_ids
                       for x, y in zip(base[srv.ipc].results, after.results))
            res.add(f"{srv.case.name}_probe_unchanged", after.all_ok and same)
            res.add(f"{srv.case.name}_alive", host.alive(srv.pid))
        return res
    except Exception as e:
        res.add("elastic_execution", False, str(e))
        return res
    finally:
        if usage is not None:
            usage.stop()
            res.add("memory_sampling", usage.error is None, str(usage.error or ""))
            if budget is not None:
                try:
                    observed = [s for t, _, s in usage.samples if t >= budget_start]
                    within_budget = bool(observed) and all(budget.within_budget(s) for s in observed)
                    res.add("shared_budget_enforced", within_budget,
                            f"{len(observed)} samples; shared cap {budget_mib} MiB")
                except RuntimeError as e:
                    res.add("shared_budget_enforced", False, str(e))
            # Timeline in MiB; retain the active-usage series for existing consumers
            # and include exact counters to audit reservations and budget transfers.
            timeline = [{"t": round(t, 1), "gpu": g,
                         **{srv.case.name: usage.of(srv, u) >> 20 for _, srv in started},
                         "segments": {k: v for _, srv in started
                                      for k, v in server_segments(srv, u).items()}}
                        for t, g, u in usage.samples]
            (ct_a.out_dir / f"{pair.name}-usage.json").write_text(json.dumps(timeline))
        for host, srv in started:
            stop_and_check(host, srv, res, prefix=f"{srv.case.name}_")
        if ballast is not None:
            ct_a.stop_server(ballast)
        probe.close()
        check_gpu_released(res, ct_a.gpus)
