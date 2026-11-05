"""Time an eager GRUCell loop on an Arc A770 while extra device memory is held: in many
allocations, or the same bytes in one. Also times the loop's graph replay when capture is
enabled (SYCL_UR_USE_LEVEL_ZERO_V2=1 NEOReadDebugKeys=1 OverrideRecordReplayGraphCapability=1);
without those settings the replay column is left out.

    python allocation_cost.py [width] [batch] [steps]

The extra memory is held as a number of 1 MiB tensors. PyTorch's caching allocator places two
of them in each 2 MiB segment it obtains from the driver, so the segment count it reports is
the number of device allocations the driver sees. With `expandable_segments:True` the allocator
maps its memory into one growing range instead and reports no segments.
"""
import sys
import time

import torch

width, batch, steps = (int(a) for a in (sys.argv[1:] + ["512", "32", "64"][len(sys.argv) - 1:]))
torch.manual_seed(0)
cell = torch.nn.GRUCell(width, width).to("xpu")
xs = torch.randn(steps, batch, width, device="xpu")
h0 = torch.randn(batch, width, device="xpu")


def loop():
    h = h0
    for t in range(steps):
        h = cell(xs[t], h)
    return h


def per_step(f, repeats=20):
    f()
    torch.xpu.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        f()
    torch.xpu.synchronize()
    return (time.perf_counter() - start) / repeats / steps * 1e6


def segments():
    return torch.xpu.memory_stats().get("segment.all.current", 0)


def record():
    side = torch.xpu.Stream()
    side.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(side):
        loop()
    torch.xpu.current_stream().wait_stream(side)
    graph = torch.xpu.XPUGraph()
    try:
        with torch.xpu.graph(graph):
            loop()
    except RuntimeError:
        return None
    return graph.replay


def held(n):
    return [torch.empty(2**18, device="xpu") for _ in range(n)]


def measure(label, ballast, replay):
    row = f"{label:<34} {segments():>8} {torch.xpu.memory_reserved() / 2**30:>8.2f} {per_step(loop):>9.0f}"
    if replay:
        row += f" {per_step(replay):>9.0f}"
    print(row, flush=True)
    del ballast[:]
    torch.xpu.empty_cache()


with torch.no_grad():
    replay = record()
    print(f"width {width}, batch {batch}, {steps} steps; microseconds per step")
    print(f"{'extra memory':<34} {'segments':>8} {'GiB':>8} {'eager':>9}" + (f" {'replay':>9}" if replay else ""))
    for n in (0, 250, 500, 1000, 2000, 4000, 0):
        ballast = held(n)
        nbytes = n * 2**20
        measure(f"{n} tensors of 1 MiB", ballast, replay)
        if n == 4000:
            one = [torch.empty(nbytes // 4, device="xpu")]
            measure(f"{nbytes / 2**30:.2f} GiB in one allocation", one, replay)
