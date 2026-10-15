# HLB-CIFAR10 / Flow / Apple MLX — evidence and parity contract

This is a **versioned evidence register**, not a claim that all upstream targets
have been met. Independent Flow and MLX implementations make it possible to
measure where a training method matches exactly, approximately, or not at all.

## Established results (15 October 2026)

| Gate | Evidence | Result |
| --- | --- | --- |
| Flow f32 convolution / gradients vs independent CPU reference | [Ubuntu native tests](https://github.com/godofecht/hlb-cifar-flow/actions/runs/38766125220) | Passed; the optimised convolution was 3.80x faster in one microbenchmark (9.309 vs 2.449 ms across eight calls) |
| Flow vs MLX forward logits, same checkpoint and synthetic 2-sample tensor | [M1 parity](https://github.com/godofecht/hlb-cifar-flow/actions/runs/38770660041) | Passed; max absolute discrepancy **3.725290298e-9** |
| Flow hand-coded loss/gradients vs MLX autograd | [M1 parity](https://github.com/godofecht/hlb-cifar-flow/actions/runs/38770660041) | Passed; 12 checked values; max absolute discrepancy **5.781650543e-6** |
| Apple MLX autograd fine-tuning, versioned checkpoint reloading | [M1 training / export](https://github.com/godofecht/hlb-cifar-flow/actions/runs/38770346755) | Passed on tiny synthetic CIFAR-format records |
| Authentic data source ingestion, Flow training and checkpoint | [4-way Linux data matrix](https://github.com/godofecht/hlb-cifar-flow/actions/runs/38770140862) | Passed on Fashion-MNIST, spoken digits, UCI HAR, AG News |
| Native Flow vs MLX predictions using the same checkpoint and actual dataset records | [4-way M1 model comparison](https://github.com/godofecht/hlb-cifar-flow/actions/runs/38770896284) | **80/80 identical class predictions** (20 per dataset), four modalities |

### What the real-data 80/80 test actually demonstrates

Two independent runtimes — the self-hosted Flow compiler producing native
code, and Apple MLX producing Apple Silicon GPU computations — agree on each
prediction given the **same learned weights and preprocessing**. The 80 real
samples were held-out test examples, not the tiny synthetic fixture.

This is **classification-decision parity** (and a separate seeded synthetic
experiment verifies floating-point logit and gradient parity). It is not
statistical evidence of generalisation or proof of state-of-the-art accuracy.

The accuracy observed in that intentionally tiny, one-epoch, width-two trial
was Fashion-MNIST 0%, spoken digits 10%, UCI HAR 20%, and AG News 25%.
Neither the Flow nor MLX implementation has been qualified for task accuracy
on those four datasets. The 10-logit head stays in place for all four,
including the 6-class UCI HAR and 4-class AG News tasks.

### Independence and leakage

Adapters use documented source splits. Fashion-MNIST uses published train/test;
FSDD holds recording indices 0–4 out as specified by its maintainers; UCI HAR
uses the original subject-disjoint training/test split; AG News retains the
original CSV split. Train-only channel normalisation and PCA whitening are not
fit on the test images. The training-only holdout is taken from the training
subset; when `HLB_VAL_HOLDOUT` is enabled the official test split is not
loaded by `train.flow`.

**Image augmentation is not automatically safe for other modalities.** Set
`HLB_DISABLE_SPATIAL_AUGMENT=1` when learning from speech spectrograms,
inertial signals, or text byte grids. Both multimodal workflows do this.

### How to reproduce

Native Flow correctness:

```bash
flow run smoke.flow
flow run bench.flow
```

Create genuine source data and train a small experimental model (example:
spoken digits):

```bash
python -m pip install numpy
python scripts/convert_modality.py --dataset spoken-digits --train 40 --test 20
HLB_DISABLE_SPATIAL_AUGMENT=1 HLB_TRAIN_LIMIT=40 HLB_VAL_HOLDOUT=8 HLB_WIDTH=2 HLB_BATCH=4 HLB_EPOCHS=1 flow run train.flow
HLB_EVAL_LIMIT=20 HLB_WIDTH=2 HLB_BATCH=4 HLB_DUMP_PREDICTIONS=1 flow run infer.flow > flow-infer.txt
```

On Apple Silicon with macOS 14+:

```bash
python -m pip install mlx numpy
python backends/mlx_backend.py infer --checkpoint hlb-ema.bin --limit 20 --batch 4 > mlx-infer.txt
python scripts/compare_inference.py flow-infer.txt mlx-infer.txt --dataset spoken-digits
```

The [MLX parity workflow](../.github/workflows/mlx.yml) runs Flow
`parity.flow` and `grad_parity.flow`, checks autograd parity, tests native
MLX weight optimisation and verifies Flow re-import of the MLX-trained
checkpoint. The [four-modality Apple MLX workflow](../.github/workflows/mlx-modalities.yml)
tests actual source records, exact predictions and bidirectional checkpoint
exchange.

The MLX Python implementation is a **separate Apple MLX backend**: the
native Flow reference does not currently invoke MLX through a Flow FFI.
It uses MLX kernels and MLX automatic differentiation, not PyTorch.

## Unresolved gaps: no claim of upstream HLB-CIFAR10 parity yet

The upstream reference [tysam-code/hlb-CIFAR10](https://github.com/tysam-code/hlb-CIFAR10)
historically reported 94% in under 6.3 seconds on an NVIDIA A100 with FP16
and tensor-core optimisations. The following must be achieved to make a
scientifically defensible *task-and-speed parity* claim:

| Acceptance dimension | Threshold / obligation | Current state |
| --- | --- | --- |
| CIFAR-10 generalisation | Full 50k train, holdout for tuning, 10k untouched final test, repeated seeds | Not measured |
| Architecture/schedule | HLB channel geometry, correct whitening, batch 1024, ~12.1 epochs, EMA write-back, augmentation and optimisers | Partially implemented; CPU/MLX extensions differ from upstream |
| Forward and backward numeric | Matched weights/data, per-layer outputs and gradients including optimised layers | Sampled end-to-end and unit tests pass; broader tensor/dtype matrix not yet done |
| GPU acceleration | Measure on actual Apple Silicon and same-device baselines; compare device/system properties | MLX supported, no qualified throughput result |
| Framework comparison | Upstream PyTorch on same device and identical dataset, precision and accounting | Not done |
| Modality accuracy | Accuracy/confusion matrices, confidence intervals vs chance + appropriate task-specific baselines | Not done |
| Precision parity | f32/f16/bf16 numerical error bounds and gradient stability | f32 only |
| Timing parity | Data, whitening, compilation, train, validation, checkpoint, energy/memory reported separately | Partial; no full benchmark |

Avoid using "feature parity" to mean only "imports and runs." Pass the target
accuracy, stability and end-to-end training gates before marking upstream
equivalence achieved. The next implementation milestone is tracked in
[GPU training issue #1](https://github.com/godofecht/hlb-cifar-flow/issues/1).
