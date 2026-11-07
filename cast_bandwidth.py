"""Bandwidth of dtype casts and plain copies on one card (README section 7). Autocast training
spends much of its device time in float32 <-> bfloat16 casts; on torch 2.15's XPU build those run
as an unrolled elementwise kernel at half the card's bandwidth.

    python cast_bandwidth.py DEVICE    # xpu (with the settings of sections 4 and 5) or cuda

GB/s counts bytes read plus bytes written. One JSON line per case.
"""
import json
import sys

import torch

dev = torch.device(sys.argv[1])
api = getattr(torch, dev.type)
n = 4096 * 1536  # one GPT-small activation, 6.3M elements
x32 = torch.randn(n, device=dev)
x16 = x32.to(torch.bfloat16)
t32 = torch.randn(4096, 1536, device=dev).t()  # transposed view: a strided source
cases = {
    "copy fp32 -> fp32": (lambda: x32.clone(), 8),
    "copy bf16 -> bf16": (lambda: x16.clone(), 4),
    "cast fp32 -> bf16": (lambda: x32.to(torch.bfloat16), 6),
    "cast bf16 -> fp32": (lambda: x16.to(torch.float32), 6),
    "cast fp32 -> bf16, strided source": (lambda: t32.to(torch.bfloat16), 6),
    "contiguous fp32, strided source": (lambda: t32.contiguous(), 8),
}
for name, (fn, bytes_per) in cases.items():
    for _ in range(10):
        fn()
    s, e = api.Event(enable_timing=True), api.Event(enable_timing=True)
    api.synchronize()
    s.record()
    for _ in range(200):
        fn()
    e.record()
    api.synchronize()
    us = s.elapsed_time(e) * 1000 / 200
    print(json.dumps({"device": dev.type, "case": name, "us": round(us, 1), "GB_per_s": round(n * bytes_per / us / 1e3, 1)}), flush=True)
