"""torch._foreach_norm, which clip_grad_norm_ uses by default, against a norm taken tensor by tensor."""
import sys

import torch

device = sys.argv[1] if len(sys.argv) > 1 else "xpu"
torch.manual_seed(0)
shapes = [(107805, 192), (3072, 1024), (3072, 192), (1024, 1024), (512, 64), (64,), (1,), (3072,), (256, 768)]
for copies in (1, 12):
    grads = [torch.randn(s, device=device) for s in shapes * copies]
    one_by_one = torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])).item()
    multi = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads, 2.0))).item()
    params = [torch.nn.Parameter(torch.zeros_like(g)) for g in grads]
    for p, g in zip(params, grads):
        p.grad = g
    clipped = torch.nn.utils.clip_grad_norm_(params, float("inf")).item()
    on_cpu = torch.linalg.vector_norm(torch.stack([g.cpu().norm() for g in grads])).item()
    print(f"torch {torch.__version__} {device}, {len(grads)} tensors: one by one {one_by_one:.1f}, "
          f"_foreach_norm {multi:.1f}, clip_grad_norm_ {clipped:.1f}, CPU {on_cpu:.1f}")
