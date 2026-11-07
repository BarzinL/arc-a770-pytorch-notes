"""Which scaled-dot-product-attention backends run forward and backward on one card, how fast,
and whether they agree with the math backend (README section 7). Shape: gpt_bench.py's small GPT
attention, batch 16, 6 heads, 256 tokens, head 64, causal, bf16.

    python sdpa_backends.py DEVICE     # xpu (with the settings of sections 4 and 5) or cuda
"""
import json
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

dev = torch.device(sys.argv[1])
api = getattr(torch, dev.type)
torch.manual_seed(0)
q, k, v = (torch.randn(16, 6, 256, 64, device=dev, dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
dout = torch.randn(16, 6, 256, 64, device=dev, dtype=torch.bfloat16)


def run():
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    gq, gk, gv = torch.autograd.grad(out, (q, k, v), dout)
    return out, gq


with sdpa_kernel(SDPBackend.MATH):
    ref_out, ref_gq = run()
NAMES = ("MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "OVERRIDEABLE")
for b in (getattr(SDPBackend, n) for n in NAMES if hasattr(SDPBackend, n)):
    row = {"device": dev.type, "torch": torch.__version__, "backend": b.name}
    try:
        with sdpa_kernel(b):
            out, gq = run()
            for _ in range(5):
                run()
            s, e = api.Event(enable_timing=True), api.Event(enable_timing=True)
            api.synchronize()
            s.record()
            for _ in range(50):
                run()
            e.record()
            api.synchronize()
        row["us_fwd_bwd"] = round(s.elapsed_time(e) * 1000 / 50, 1)
        row["max_err_out"] = round(float((out.float() - ref_out.float()).abs().max()), 4)
        row["max_err_grad_q"] = round(float((gq.float() - ref_gq.float()).abs().max()), 4)
    except Exception as exc:  # a backend the card or build does not offer
        row["error"] = str(exc).splitlines()[0][:120]
    print(json.dumps(row), flush=True)
