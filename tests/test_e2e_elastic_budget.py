# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0
"""Physical reservations must drain before a shared KV allowance is transferred."""

from types import SimpleNamespace

import pytest

from e2e import elastic


def test_segments_include_zero_limits_and_reservations(tmp_path):
    (tmp_path / "owned").write_bytes(elastic.SEGMENT.pack(0, 128, 64))
    (tmp_path / "invalid").write_bytes(elastic.SEGMENT.pack(-1, 0, 0))
    (tmp_path / "unrelated").write_text("other IPC")
    assert elastic.read_segments(str(tmp_path)) == {
        "owned": {"limit": 0, "used": 128, "preallocated": 64}}


def test_usage_matches_only_owned_groups():
    srv = SimpleNamespace(ipc="owned")
    segments = {name: {"used": value} for name, value in
                [("owned", 1), ("owned_g1", 2), ("owned_garbage", 4), ("owned_other", 8)]}
    assert elastic.Usage.of(srv, segments) == 3


def test_transfer_shrinks_donor_before_expanding_recipient(monkeypatch):
    states = {name: {"limit": limit, "used": 10, "preallocated": 5}
              for name, limit in [("a", 20), ("b", 80)]}
    servers = [(None, SimpleNamespace(ipc=name)) for name in states]
    budget = elastic.SharedBudget(servers, 100)
    budget.established = True
    monkeypatch.setattr(elastic, "read_segments", lambda: states)
    changes = []

    def apply(host, srv, target, previous):
        changes.append(srv.ipc)
        states[srv.ipc]["limit"] = target

    monkeypatch.setattr(budget, "_set", apply)
    budget.set_shares((80, 20), "transfer")
    assert changes == ["b", "a"]
    assert budget.history[0]["stage"] == "transfer"
    states["b"]["preallocated"] = 90
    assert not budget.within_budget(states)


def test_successful_limit_command_is_not_evidence_of_release(monkeypatch):
    srv = SimpleNamespace(ipc="a")
    host = SimpleNamespace(exec=lambda _: SimpleNamespace(returncode=0, stderr=b""))
    budget = elastic.SharedBudget([(host, srv)], 100, timeout=0)
    monkeypatch.setattr(elastic, "read_segments", lambda: {
        "a": {"limit": 0, "used": 0, "preallocated": 1}})
    with pytest.raises(RuntimeError, match="did not settle"):
        budget._set(host, srv, 0, 100)


def test_shrink_waits_for_reserved_pages(monkeypatch):
    srv = SimpleNamespace(ipc="a")
    host = SimpleNamespace(exec=lambda _: SimpleNamespace(returncode=0, stderr=b""))
    budget = elastic.SharedBudget([(host, srv)], 100)
    states = iter([{"a": {"limit": 0, "used": 0, "preallocated": 1}},
                   {"a": {"limit": 0, "used": 0, "preallocated": 0}}])
    sleeps = []
    monkeypatch.setattr(elastic, "read_segments", lambda: next(states))
    monkeypatch.setattr(elastic.time, "sleep", sleeps.append)
    budget._set(host, srv, 0, 100)
    assert sleeps == [0.25]


def test_missing_or_multiple_segments_fail_closed():
    srv = SimpleNamespace(ipc="a")
    for segments in ({}, {"a": {}, "a_g1": {}}):
        with pytest.raises(RuntimeError, match="one live segment"):
            elastic.SharedBudget.state(srv, segments)
