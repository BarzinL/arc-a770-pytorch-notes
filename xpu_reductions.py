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


class _Stock(torch.autograd.Function):
    """The stock kernel, called below the autograd key (calling it at the key would reach
    these replacements again), with the gradient that sum and mean have. Without this
    wrapper the result carries no gradient: every trailing reduction would return the right
    value and silently cut training's gradient."""

    @staticmethod
    def forward(ctx, x, dim, dims, keepdim, dtype, mean):
        ctx.shape, ctx.dims, ctx.keepdim, ctx.in_dtype, ctx.mean = x.shape, dims, keepdim, x.dtype, mean
        with torch._C._AutoDispatchBelowAutograd():
            if mean:
                return torch.ops.aten.mean.dim(x, dim, keepdim, dtype=dtype)
            return torch.ops.aten.sum.dim_IntList(x, dim, keepdim, dtype=dtype)

    @staticmethod
    def backward(ctx, grad):
        if not ctx.keepdim:
            for d in ctx.dims:
                grad = grad.unsqueeze(d)
        grad = grad.expand(ctx.shape)
        if ctx.mean:
            n = 1
            for d in ctx.dims:
                n *= ctx.shape[d]
            grad = grad / n
        return grad.to(ctx.in_dtype), None, None, None, None, None


def _sum(x, dim=None, keepdim=False, dtype=None):
    dims = _reduced(x, dim)
    if _stock(x, dims):
        return _Stock.apply(x, dim, dims, keepdim, dtype, False)
    return _sum_by_product(x, dims, keepdim, dtype)[0]


def _mean(x, dim=None, keepdim=False, dtype=None):
    dims = _reduced(x, dim)
    if _stock(x, dims):
        return _Stock.apply(x, dim, dims, keepdim, dtype, True)
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
    # Gradients through every form of reduction must match the CPU's.
    bad = []
    for op in ("sum", "mean"):
        for dim in ([0], [-1], [1, 2], None):
            x = torch.randn(64, 32, 16)
            w = torch.randn_like(getattr(x, op)(dim))
            grads = []
            for dev in ("cpu", "xpu"):
                t = x.detach().clone().to(dev).requires_grad_()
                (getattr(t, op)(dim) * w.to(dev)).sum().backward()
                grads.append(None if t.grad is None else t.grad.cpu())
            if grads[1] is None or not torch.allclose(grads[0], grads[1], atol=1e-5):
                bad.append(f"{op}({dim})")
    print("gradients that differ from the CPU's:", bad or "none")
