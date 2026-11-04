"""Record a stepped GRUCell loop with torch.xpu.XPUGraph on an Arc A770 and replay it.

Run with SYCL_UR_USE_LEVEL_ZERO_V2=1 NEOReadDebugKeys=1 OverrideRecordReplayGraphCapability=1;
without them capture is refused.

    python capture_gru.py [width] [batch] [steps]
"""
import sys
import time

import torch

width, batch, steps = (int(a) for a in (sys.argv[1:] + ["1024", "64", "64"][len(sys.argv) - 1:]))
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


with torch.no_grad():
    eager = loop()
    side = torch.xpu.Stream()
    side.wait_stream(torch.xpu.current_stream())
    with torch.xpu.stream(side):
        loop()
    torch.xpu.current_stream().wait_stream(side)
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        recorded = loop()
    graph.replay()
    torch.xpu.synchronize()
    difference = (recorded - eager).abs().max().item()
    xs.add_(0.5)
    graph.replay()
    followed = (recorded - loop()).abs().max().item()
    print(f"width {width}, batch {batch}, {steps} steps: eager {per_step(loop):.0f} us/step, "
          f"replay {per_step(graph.replay):.0f} us/step, replay vs eager {difference}, "
          f"after changing the input in place {followed}")
