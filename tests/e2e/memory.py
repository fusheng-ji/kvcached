# SPDX-License-Identifier: Apache-2.0
"""GPU memory sampling, including unified-memory devices without NVML counters."""
from __future__ import annotations

import ctypes as C
import subprocess
import threading

MIB = 1 << 20


def smi(gpu: str, fields: str) -> str:
    return subprocess.check_output(
        ['nvidia-smi', '-i', gpu, f'--query-gpu={fields}', '--format=csv,noheader,nounits'],
        text=True, timeout=10).strip()


def parse_memory(text: str) -> tuple[int, int]:
    used, total = (int(x.strip()) for x in text.split(','))
    if not 0 <= used < total:
        raise ValueError(f'invalid GPU memory counters: {text}')
    return used, total


class CudaMemory:
    """One persistent primary context on the selected host GPU, not CUDA ordinal 0."""

    def __init__(self, gpu: str):
        self.lib = C.CDLL('libcuda.so.1')
        self.lock = threading.Lock()
        self.context = C.c_void_p()
        self.device = C.c_int()
        self.check(self.lib.cuInit(0))
        # PCI identity avoids host index / CUDA_VISIBLE_DEVICES remapping errors.
        bus = smi(gpu, 'pci.bus_id').lower()
        domain, busno, slot = bus.split(':')
        bus = f'{int(domain, 16):04x}:{busno}:{slot}'
        self.check(self.lib.cuDeviceGetByPCIBusId(C.byref(self.device), bus.encode()))
        self.check(self.lib.cuDevicePrimaryCtxRetain(C.byref(self.context), self.device))

    @staticmethod
    def check(result: int):
        if result:
            raise RuntimeError(f'CUDA memory probe failed with driver error {result}')

    def read(self) -> tuple[int, int]:
        with self.lock:
            self.check(self.lib.cuCtxSetCurrent(self.context))
            free, total = C.c_size_t(), C.c_size_t()
            self.check(self.lib.cuMemGetInfo_v2(C.byref(free), C.byref(total)))
            if not 0 < free.value <= total.value:
                raise RuntimeError('CUDA returned invalid memory counters')
            return (total.value - free.value) // MIB, total.value // MIB

    def close(self):
        if self.context:
            self.check(self.lib.cuDevicePrimaryCtxRelease_v2(self.device))
            self.context = C.c_void_p()


class MemoryProbe:
    def __init__(self, gpu: str):
        self.gpu = gpu
        self.cuda = None
        try:
            parse_memory(smi(gpu, 'memory.used,memory.total'))
            self.source = 'nvidia-smi'
        except (ValueError, subprocess.SubprocessError):
            self.cuda = CudaMemory(gpu)
            self.source = 'cuda.cuMemGetInfo'
        try:
            self.read()  # Fail before launching a sampling thread.
        except Exception:
            self.close()
            raise

    def read(self) -> tuple[int, int]:
        return self.cuda.read() if self.cuda else parse_memory(smi(self.gpu, 'memory.used,memory.total'))

    def close(self):
        if self.cuda:
            self.cuda.close()
