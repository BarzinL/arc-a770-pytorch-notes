# Training on an Intel Arc A770 with PyTorch XPU: two wrong-result bugs, graph capture, and the cost of allocations

These notes come from training recurrent models on an Arc A770 (16 GB) with PyTorch's XPU
backend. Two bugs give silently wrong gradients in ordinary training code. Neither raises an
error or a warning, and each has a workaround of a few lines. The third section shows how to
turn on graph capture (`torch.xpu.XPUGraph`), which the card refuses by default. The fourth
shows that every device allocation a process holds makes each eager operation slower.

Measured on torch 2.15.0+xpu and the 2.16.0.dev20261102+xpu nightly, intel-compute-runtime
26.35.39758, Linux 7.2. Every number below comes from the scripts in this folder, run on that
card. The results are the same on both torch versions unless a table says otherwise.

## 1. Column sums and means are often wrong

A floating-point `sum` or `mean` over the leading dimension of a tensor with tens of thousands
of rows is often wrong. Most errors are a few percent of the result's norm; some are thousands
of times the correct value. Autograd's bias gradient for `nn.Linear` is such a sum, so a model
whose linear layers see tens of thousands of rows per batch gets wrong bias gradients on some
steps. In training this showed up as occasional gradient-norm spikes (a norm of 66 to 76 against
a median of 15) that disappeared when the same step was recomputed.

Reported upstream as [intel/torch-xpu-ops#6033](https://github.com/intel/torch-xpu-ops/issues/6033).

`sum_repro.py` sums 200 random float32 tensors of 19,200 to 48,576 rows and 256 columns and
compares each against a float64 sum on the CPU, counting a result as wrong when its relative
error exceeds 1e-4:

| torch | `x.sum(0)` wrong | `ones(1, rows) @ x` wrong |
|---|---|---|
| 2.15.0+xpu | 104 of 200 | 0 of 200 |
| 2.16.0.dev20261102+xpu | 98 of 200 | 0 of 200 |

What else was measured:

- `mean(0)`, a sum over two leading dimensions (`x.view(64, -1, 256).sum((0, 1))`) and
  `amax(0)` fail in the same way. A sum along the last dimension (`x.sum(1)` on a
  `(256, rows)` tensor) and a sum over the whole tensor were right in every trial.
- With 1 or 16 columns the sum was essentially never wrong. With 64, 256 and 1,024 columns it
  failed. No failure was seen below about 20,000 rows, but small sizes were sampled sparsely.
- The first reduction after other work on the card is the one that tends to go wrong. Summing
  the same resident tensor 50 times in a row gave one consistent, correct answer.
- Disabling the copy engine (`UR_L0_USE_COPY_ENGINE=0`), synchronizing after the upload, and
  filling free device memory with garbage or zeros beforehand made no difference.

### Workaround

`xpu_reductions.py` registers replacements for `aten::sum.dim_IntList` and `aten::mean.dim` at
the `AutogradXPU` dispatch key. Reductions over the trailing dimensions, over the whole tensor,
or of integer tensors still go to the stock kernel. Every other floating-point reduction is
computed as a matrix product with a row of ones. Because the replacement sits at the autograd
key, the sums that autograd's backward formulas call, such as `nn.Linear`'s bias gradient, go
through it as well.

```python
import xpu_reductions
xpu_reductions.install()   # once, before training
```

Correction (2026-11-04): the first published version of `xpu_reductions.py` sent reductions
over trailing dimensions to the stock kernel below the autograd key with no backward, so
`sum(-1)`, `mean(-1)` and reductions over the whole tensor returned the right values but no
gradient, and training through them silently lost part of its gradient. The current version
wraps that call in an autograd function with the gradients sum and mean have, and the script
now also checks gradients of every reduction form against the CPU. If you copied the earlier
version, replace it.

Running `python xpu_reductions.py` measures the stock kernels and
`python xpu_reductions.py --installed` the replacement. Sum and mean take turns as the first
reduction after each upload, 15 trials each; the bias gradient is checked 30 times:

| torch | Stock: `sum(0)`, `mean(0)`, bias gradient | Installed |
|---|---|---|
| 2.15.0+xpu | 10, 6, 17 wrong | 0, 0, 0 |
| 2.16.0.dev20261102+xpu | 8, 4, 18 wrong | 0, 0, 0 |

In a recurrent model's training run, the steps with a wrong on-card bias sum went from 13 and
14 of 60 (worst 11,561 times off) to 0 of 80 and the gradient-norm spikes stopped. With the
corrected version, every gradient of a training step equals the CPU's to within 6e-6 relative,
and a step takes 2.69 s against 2.70 s before. `amax` and `amin` over leading dimensions are not
covered; replace them the same way if your model uses them.

## 2. `clip_grad_norm_` overcounts the gradient norm

`torch.nn.utils.clip_grad_norm_` computes the total norm with the multi-tensor kernel
`torch._foreach_norm` by default. On the XPU that kernel returns a total larger than the true
norm, so every clipped step is scaled down more than it should be, and every logged gradient
norm is too large. `foreach_norm_repro.py` compares it with norms taken one tensor at a time:

| Tensors | One at a time | `_foreach_norm` and `clip_grad_norm_` | CPU |
|---|---|---|---|
| 9 | 5070.5 | 7170.3 | 5066.0 |
| 108 | 17566.0 | 24844.8 | 17550.4 |

Taken tensor by tensor, every result is wrong: about 1.414 (the square root of 2) times too large
for tensors of tens of thousands to millions of elements, exactly 2 times for tensors of 1 and
64 elements, and 1.64 times for one of 3,072, every time, with a single tensor in the list as well as with many. For a real model's
gradients the logged norm was 23.1 where the true norm was 15.5. The same script on a CUDA card
(an RTX 3060 Ti) gives agreeing norms, so the defect is specific to the XPU kernel. It is
present with or without the graph-capture settings below.

Reported upstream as [intel/torch-xpu-ops#6035](https://github.com/intel/torch-xpu-ops/issues/6035).

### Workaround

Pass `foreach=False` on the XPU:

```python
torch.nn.utils.clip_grad_norm_(params, max_norm, foreach=False)
```

## 3. Graph capture on the A770

`torch.xpu.XPUGraph` refuses to capture on this card: "Device does not support graph record and
replay feature (UR_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP)". There are two causes, and
both settings below are needed:

- The SYCL runtime's v1 Level Zero adapter, the default for this card's generation, reports no
  graph support without asking the driver. `SYCL_UR_USE_LEVEL_ZERO_V2=1` selects the v2
  adapter, which asks.
- The driver advertises the `ZE_experimental_record_replay_graph` extension but reports no
  graph modes for the A770. `NEOReadDebugKeys=1 OverrideRecordReplayGraphCapability=1` makes it
  report immutable graphs.

```sh
SYCL_UR_USE_LEVEL_ZERO_V2=1 NEOReadDebugKeys=1 OverrideRecordReplayGraphCapability=1 python capture_gru.py
```

`capture_gru.py` records a 64-step `nn.GRUCell` loop and replays it:

| Width, batch | Eager, per step | Replay, per step | Replay against eager |
|---|---|---|---|
| 1,024, 64 | 173 microseconds | 111 | difference 0.0 |
| 512, 32 | 176 microseconds | 21 | difference 0.0 |

The replay also matches eager after the input tensor is changed in place, so it reads inputs'
current values. Eager costs about 170 microseconds per step at both widths, which is the launch
cost; the replay removes it, and at width 1,024 what remains is the arithmetic.

Backward passes and optimizer steps capture as well. A whole training step of the same loop
(forward, backward, and SGD or Adam with `capturable=True`) replayed 6 to 14 times faster than
eager, with losses and parameters equal to eager's to the last bit over 20 steps. In a larger
model, recording its 96-step recurrent loop forward and backward cut a training step from 6.3 s
to 2.7 s.

Caveats:

- Intel does not enable this capability on the A770 by default, so it is not validated on this
  card. The checks here (exact agreement with eager, inputs followed, training steps equal to
  eager) cover the operations these models use, and no more.
- An embedding's backward could not be captured (an unknown Level Zero error). Operations whose
  output size depends on the data, such as masked indexing, cannot be captured either, as with
  CUDA graphs; they have to run outside the recorded region.
- The two bugs above happen with or without capture.

## 4. Every allocation held makes each eager operation slower

The host time of an eager operation grows with the number of device allocations the process
holds, not with the amount of memory. `allocation_cost.py` times an eager `nn.GRUCell` loop
(width 512, batch 32) while holding a number of 1 MiB tensors, which PyTorch's caching
allocator places two to a 2 MiB segment, so each pair is one device allocation; then the same
bytes in one tensor:

| Extra memory held | Allocator segments | Eager, microseconds per step | Replay, microseconds per step |
|---|---|---|---|
| none | 3-4 | 167-191 | 21 |
| 250 tensors of 1 MiB | 127-129 | 300-318 | 21 |
| 500 | 252-254 | 437-457 | 21 |
| 1,000 | 502-504 | 740-758 | 37 |
| 2,000 | 1,002-1,004 | 1,484-1,494 | 35 |
| 4,000 | 2,002-2,004 | 3,325-3,329 | 39 |
| 3.91 GiB in one tensor | 3-5 | 171-189 | 21 |

The ranges span two runs: the default runtime, and the settings of section 3 (the replay column
is from that run only). Each step costs about 1.1 microseconds more per extra allocation at
first and about 1.6 at 2,000. A recorded graph's replay grows far less, from 21 to 39
microseconds.

- `expandable_segments:True` in `PYTORCH_ALLOC_CONF` does not help: the same tensors cost the
  same (169 rising to 3,241 microseconds per step), and 3.91 GiB in one tensor cost 363
  microseconds per step instead of 171, presumably because the allocator builds it from many
  physical chunks. With this setting PyTorch reports no segment count.
- Why each allocation costs every operation was not measured. The card runs on the `i915`
  kernel driver here. One possibility is that the runtime passes every allocation to the
  driver at each submission.

In a training job this is easy to hit, because a real model's working memory is spread over
hundreds of segments. A recurrent model whose training step dispatches about 36,000 eager
operations held about 510 segments and took 2.8 s per step; recording a second part of it with
its own `make_graphed_callables` call added about 270 segments (each call gets a private memory
pool) and made the step 1.4 s slower, about 5 ms per segment. Recording both parts in one call,
which gives them one shared pool, cut the extra segments to about 70 and the step to 3.2 s:

```python
# one pool for both recordings; replay them in this order, backward in reverse
first, second = torch.xpu.make_graphed_callables((first, second), (first_args, second_args))
```

Two practical consequences: keep the number of live allocations low (few large tensors rather
than many small ones, one shared pool for all recordings), and expect eager timings on this card
to depend on everything else the process holds. These measurements are on torch 2.15.0+xpu only.

## Files

| File | What it does |
|---|---|
| `sum_repro.py` | The column-sum bug in 14 lines |
| `xpu_reductions.py` | The workaround for sum and mean; run directly, it measures before and after |
| `foreach_norm_repro.py` | The gradient-norm bug |
| `capture_gru.py` | Graph capture of a GRU loop, timed and checked against eager |
| `allocation_cost.py` | Eager and replayed step time against the number of allocations held |
