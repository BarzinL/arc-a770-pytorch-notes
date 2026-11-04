"""Workaround for wrong column sums and means on the Intel XPU (intel/torch-xpu-ops#6033).

Call install() once, before training. It replaces aten::sum.dim_IntList and aten::mean.dim
for XPU tensors. Reductions over the trailing dimensions, over the whole tensor, or of
integer tensors go to the stock kernel; every other floating-point reduction is computed
as a matrix product with a row of ones. It is registered at the AutogradXPU key, so the
sums that autograd's backward formulas call, such as nn.Linear's bias gradient, go through
it too. amax and amin are not covered.
"""
import torch

_library = None


def _reduced(x, dim):
    return list(range(x.dim())) if not dim else sorted({d % x.dim() for d in dim})


def _stock(x, dims):
    return not x.is_floating_point() or dims == list(range(x.dim() - len(dims), x.dim()))


def _sum_by_product(x, dims, keepdim, dtype):
    kept = [d for d in range(x.dim()) if d not in dims]
    n = 1
    for d in dims:
        n *= x.shape[d]
    y = x.permute(dims + kept).reshape(n, -1)
    if dtype is not None:
        y = y.to(dtype)
    out = (y.new_ones(1, n) @ y).reshape([x.shape[d] for d in kept])
    if keepdim:
        for d in dims:
            out = out.unsqueeze(d)
    return out, n


def _sum(x, dim=None, keepdim=False, dtype=None):
    dims = _reduced(x, dim)
    if _stock(x, dims):
        with torch._C._AutoDispatchBelowAutograd():
            return torch.ops.aten.sum.dim_IntList(x, dim, keepdim, dtype=dtype)
    return _sum_by_product(x, dims, keepdim, dtype)[0]


def _mean(x, dim=None, keepdim=False, dtype=None):
    dims = _reduced(x, dim)
    if _stock(x, dims):
        with torch._C._AutoDispatchBelowAutograd():
            return torch.ops.aten.mean.dim(x, dim, keepdim, dtype=dtype)
    total, n = _sum_by_product(x, dims, keepdim, dtype)
    return total / n


def install():
    global _library
    if _library is None:
        _library = torch.library.Library("aten", "IMPL")
        _library.impl("sum.dim_IntList", _sum, "AutogradXPU")
        _library.impl("mean.dim", _mean, "AutogradXPU")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--installed":
        install()
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 256).to("xpu")
    wrong = {"sum(0)": 0, "mean(0)": 0, "bias gradient": 0}
    for i in range(30):
        rows = 64 * int(torch.randint(300, 760, ()))
        x = torch.randn(rows, 256)
        d = x.to("xpu")
        # The first reduction after other work is the one that tends to go wrong, so one per upload.
        name, got, want = (("sum(0)", d.sum(0), x.double().sum(0)) if i % 2 else ("mean(0)", d.mean(0), x.double().mean(0)))
        wrong[name] += ((got.cpu().double() - want).norm() / want.norm()).item() > 1e-4
        g = torch.randn(rows, 256)
        lin.zero_grad()
        lin(torch.randn(rows, 256, device="xpu")).backward(g.to("xpu"))
        want = g.double().sum(0)
        wrong["bias gradient"] += ((lin.bias.grad.cpu().double() - want).norm() / want.norm()).item() > 1e-4
    print(f"torch {torch.__version__}, workaround {'installed' if _library else 'not installed'}: wrong {wrong} (sum and mean of 15 each, bias gradient of 30)")
