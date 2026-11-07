"""Time a plain GPT (pre-norm transformer, PyTorch only) on one card: training steps at two sizes
and token-by-token generation at batch 1. For comparing cards, and the A770 with and without
UR_L0_USM_ALLOCATOR (section 4). Prints one JSON line per measurement.

    python gpt_bench.py DEVICE      # xpu or cuda

Inputs are random tokens: the timings do not depend on the data. Each measurement is timed after
warm-up steps, with the device waited on before the clock stops. A measurement that runs out of
memory is reported as such and the rest continue.
"""
import json
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device(sys.argv[1] if len(sys.argv) > 1 else "xpu")
sync = {"xpu": torch.xpu.synchronize, "cuda": torch.cuda.synchronize}.get(DEV.type, lambda: None)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.up, self.down = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, vocab, d, layers, heads, ctx):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))


SIZES = {  # name: vocab, width, layers, heads, context, batch
    "small": (8192, 384, 6, 6, 256, 16),
    "124M": (50304, 768, 12, 12, 512, 8),
}


def report(**row):
    print(json.dumps({"device": DEV.type, **row}), flush=True)


def train(name, dtype, steps=20, warm=3):
    vocab, d, layers, heads, ctx, batch = SIZES[name]
    torch.manual_seed(0)
    model = GPT(vocab, d, layers, heads, ctx).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    ids = torch.randint(0, vocab, (batch, ctx + 1), device=DEV)

    def step():
        with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=dtype == "bf16"):
            logits = model(ids[:, :-1])
            loss = F.cross_entropy(logits.float().view(-1, vocab), ids[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss

    for _ in range(warm):
        step()
    sync()
    t = time.perf_counter()
    for _ in range(steps):
        loss = step()
    sync()
    s = (time.perf_counter() - t) / steps
    report(test=f"train {name} {dtype}", step_s=round(s, 4), tokens_per_s=round(batch * ctx / s), loss=round(float(loss.detach()), 3))


@torch.no_grad()
def generate(name, dtype, new=128, prompt=16):
    """Batch 1, no key-value cache: each new token reruns the prefix, as a naive loop does."""
    vocab, d, layers, heads, ctx, _ = SIZES[name]
    torch.manual_seed(0)
    model = GPT(vocab, d, layers, heads, ctx).to(DEV).eval()
    ids = torch.randint(0, vocab, (1, prompt), device=DEV)

    def run():
        x = ids
        for _ in range(new):
            with torch.autocast(DEV.type, dtype=torch.bfloat16, enabled=dtype == "bf16"):
                nxt = model(x[:, -ctx:])[:, -1].argmax(-1, keepdim=True)
            x = torch.cat([x, nxt], 1)
        return x

    run()
    sync()
    t = time.perf_counter()
    run()
    sync()
    s = time.perf_counter() - t
    report(test=f"generate {name} {dtype}", tokens_per_s=round(new / s, 1), ms_per_token=round(1000 * s / new, 2))


for fn, args in ((train, ("small", "fp32")), (train, ("small", "bf16")), (train, ("124M", "bf16")),
                 (generate, ("small", "fp32")), (generate, ("124M", "bf16"))):
    try:
        fn(*args)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower() and not isinstance(e, torch.OutOfMemoryError):
            raise
        report(test=f"{fn.__name__} {' '.join(args)}", error="out of memory")
    if DEV.type != "cpu":
        getattr(torch, DEV.type).empty_cache()
