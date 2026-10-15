# HLB-CIFAR10 in Flow

**Native Flow reimplementation of a CIFAR-10 convolutional classifier and training pipeline, designed as a correctness-first baseline for a future GPU training compiler benchmark.**

Upstream research reference: [tysam-code/hlb-CIFAR10](https://github.com/tysam-code/hlb-CIFAR10) (Apache-2.0). Language: [Floooooooooooow/flow](https://github.com/flooooooooooow/flow). This repository was independently written in Flow; no PyTorch or NumPy dependency, and no external C neural-network implementation.

> **Status:** Experimental CPU reference implementation; not an A100 speed record. Numerical test coverage is provided in `smoke.flow`. The GitHub Actions workflow is the authoritative compiler/build gate.

## Evidence-backed parity: Flow CPU ↔ Apple MLX

[**Full parity evidence, reproducible commands and unresolved gaps →**](docs/PARITY.md)

On an Apple M1 GitHub Actions runner, the two implementations agreed on the
same model's deterministic forward logits to **3.73e-9 maximum absolute
difference**, and selected loss/analytic-gradient values versus MLX autodiff
to **5.78e-6**. Real-data CPU training passed on Fashion-MNIST (vision),
Free Spoken Digits (audio), UCI HAR (inertial time-series) and AG News (text).

Independent macOS MLX experiments agreed on **all 80/80 per-example predicted
classes** from the official test records (20 per modality) using the
identical Flow-trained model checkpoint; these were tiny one-epoch experiments.
This is *functional and numerical parity*, **not accuracy or speed parity**
with the upstream HLB paper.

Apple Silicon MLX implementation:
[`backends/mlx_backend.py`](backends/mlx_backend.py).
CI:
[MLX forward/gradient parity](.github/workflows/mlx.yml),
[real-modality Flow training](.github/workflows/modalities.yml), and
[Flow-vs-MLX real-dataset predictions](.github/workflows/mlx-modalities.yml).

```bash
# On macOS 14+ with Apple Silicon, after native Flow has generated a checkpoint:
python -m pip install mlx numpy
HLB_WIDTH=8 flow run parity.flow > flow-parity.txt
python backends/mlx_backend.py parity --checkpoint hlb-ema.bin --flow-logits flow-parity.txt
python backends/mlx_backend.py train --checkpoint hlb-current.bin --limit 40 --holdout 8 --epochs 1
python backends/mlx_backend.py infer --checkpoint mlx-finetuned.bin --limit 20

# MLX can also initialise PCA whitening and a new model without Flow pretraining:
python backends/mlx_backend.py init --width 8 --limit 256 --output mlx-initial.bin
python backends/mlx_backend.py train --checkpoint mlx-initial.bin --limit 256 --holdout 32 --epochs 2 --output mlx-trained.bin
```

These MLX commands require CIFAR-format records in `data/cifar-10-batches-bin/`.
For non-vision modalities, run `python scripts/convert_modality.py
--dataset spoken-digits --train 40 --test 20` (or choose one of the other
supported datasets). The converter is Python-only *preprocessing*; all native
Flow training computations are in Flow, and Apple accelerator training runs
using MLX. Image flips/crops/CutMix are disabled for non-image modalities by
setting `HLB_DISABLE_SPATIAL_AUGMENT=1`.

## Implemented

| Component | Flow implementation |
| --- | --- |
| CIFAR-10 | Toronto binary dataset loader, RGB normalization from training data |
| Data augmentation | Reflect-pad/shift crop, per-image horizontal flips, 3x3 CutMix and mixed targets |
| Whitening | Data-derived 2x2 patch covariance and 12-dimensional symmetric Jacobi eigensolver; signed PCA filter pairs |
| Convolution | F32 NCHW valid/same-padded 3x3 and 2x2 forward and dInput/dWeight backward |
| Nonlinearity | Exact erf-based GELU and analytic derivative |
| Pooling | 2x2 max-pooling, global max pooling, and scatter derivatives |
| BatchNorm | Learnable beta, fixed gamma=1, running moments, train/eval modes and exact batched derivative |
| CNN | Whitening → 3 × (Conv–Pool–BN–GELU–Conv–BN–GELU) → global max → 10-class linear |
| Initialisation | Dirac-initialised first conv; exact identity-kernel second conv |
| Objective | Soft-target cross entropy with 0.2 label smoothing |
| Optimisation | Nesterov SGD, bias/non-bias rates, L2 decay, linear one-cycle learning rate |
| Evaluation | Horizontal-flip logit averaging, separate EMA weight model |
| Verification | Finite-difference gradients, convolution reference parity, pooling adjoint, BN running variance, whitening symmetry, checkpoint roundtrip |
| Checkpoints | Versioned f32 model and train-only RGB normalisation in `hlb-current.bin` and `hlb-ema.bin` |
| Inference | Standalone `infer.flow` restores EMA checkpoint, evaluates with horizontal-flip TTA, and frees intermediate activations immediately |

The native Flow code is in [src/core.flow](src/core.flow), [src/data.flow](src/data.flow) and [src/whitening.flow](src/whitening.flow), and [src/checkpoint.flow](src/checkpoint.flow). All forward passes and gradient updates are implemented in those files.

## Run

Install Flow 2.0 or build its [self-hosted compiler](https://github.com/flooooooooooow/flow). A C11 toolchain is required. Linux and macOS are supported by Flow's stable portable-C backend.

```bash
bash scripts/download-data.sh
flow run smoke.flow
flow run train.flow
flow run infer.flow
flow run bench.flow
```

The default run deliberately uses 256 training and 256 test images, width 8, batch size 8 and 2 epochs. **This is a functional CPU development configuration, not a meaningful CIFAR-10 accuracy benchmark.** The program prints loss, accuracy, EMA accuracy and full CPU training-plus-evaluation time. It saves train and EMA checkpoints with the exact input normalisation statistics so `infer.flow` can reload them correctly.

To reserve examples from the **training split** for validation rather than re-evaluating the official test set every epoch, use `HLB_VAL_HOLDOUT=32` (or a larger holdout). The holdout is taken from the end of the loaded training subset; training statistics are computed only over the remaining examples. In this mode the official test split is not loaded. Use this mode for hyperparameter tuning.

`HLB_WHITEN_STRIDE=8` (default) samples a 4×4 grid of 2×2 patches per training image for PCA whitening; `HLB_WHITEN_STRIDE=1` uses all 31×31 overlapping patches, at substantially greater CPU cost. Accumulation is performed in f64, without materializing patch tensors.

To run the entire dataset and HLB-sized channel widths, set:

```bash
HLB_TRAIN_LIMIT=50000 \
HLB_EVAL_LIMIT=10000 \
HLB_WIDTH=64 \
HLB_BATCH=1024 \
HLB_EPOCHS=13 \
HLB_WHITEN_EXAMPLES=50000 \
flow run train.flow
```

**Warning:** this configuration is computationally prohibitive on the current reference CPU implementation. It is provided to define the workload for future optimised Flow backends, not as a recommended local run. A CUDA-capable accelerated backend is needed to compete with the upstream reference timing.

## What differs from the original

This is a ground-up implementation of the network and training mechanisms, **not a bit-for-bit reproduction of the PyTorch program**. All floating-point computation is currently F32 on CPU, rather than FP16 channels-last using NVIDIA tensor cores. It supports exhaustive all-patch whitening with stride 1 or cheaper deterministic grid sampling for smaller experiments. Unlike the upstream per-chunk eigendecomposition-and-average method, it computes a single eigendecomposition of the aggregated covariance. It uses a conventional EMA evaluation model rather than upstream's additional EMA parameter write-back. The training schedule uses an integer number of epochs, and its learning-rate update uses equivalent mean-loss scaling rather than the upstream summed-loss convention. First-convolution Dirac initialisation does not reproduce the upstream weight-statistic renormalisation.

Consequently neither the published **94% in under 6.3 seconds** nor **95.79% in roughly 110 seconds** is claimed or expected here. Those were historical upstream measurements on an A100. This repository needs matched accuracy, CUDA kernels, FP16 numerical testing, proper batching optimisation and reproducible end-to-end timing before comparing speed.

## Benchmark protocol

First verify gradient agreement on synthetic small problems with finite differences, then convergence using complete training and held-out validation. Record compiler commit, backend, device, training-data count, batch size, network width, precision, epoch count, random seed, CutMix policy and warm/cold data loading. Keep CIFAR-10's official test split out of hyperparameter selection in serious experiments.

Report data preparation, PCA whitening, training and evaluation times **separately**. The original benchmark deliberately does not include all of them in its headline training time.

CI also checks exact checkpoint roundtrips, an opt-in held-out validation partition, streaming-inference numerical parity, and standalone inference after a synthetic training run.

For genuine CIFAR-10 measurements, trigger the [manual real-data workflow](.github/workflows/real-data.yml) from GitHub Actions. It downloads the original binary archive from the University of Toronto, reserves a training-only validation holdout, trains in native Flow, and uploads the model artifacts. Its conservative CPU defaults are for correctness, **not** reproducing HLB accuracy or GPU timing. The synthetic-fixture workflow does not exercise the official image dataset.

CI additionally generates a **synthetic** CIFAR-binary fixture with `scripts/generate-fixture.py` and performs an end-to-end train/evaluate run. Those generated pixels are not part of the official CIFAR-10 dataset and must never be used for accuracy claims.

### Measured CPU kernel improvement

The Flow-native contiguous convolution kernel is checked against an independent reference implementation for forward and backward parity. In [GitHub Actions run 38766125220](https://github.com/godofecht/hlb-cifar-flow/actions/runs/38766125220), a 2×8×16×16 input with 16 output channels and 3×3 filters took **9.309 ms** with the reference and **2.449 ms** with the optimised kernel over eight iterations (**3.80×** reference/optimised). The reported maximum absolute output error was 0.0 at seven decimal places. Results are hardware/compiler dependent and **are not an end-to-end training speedup**. Run `flow run bench.flow` for repeatable local measurements.

The immediate performance roadmap is batched GEMM/im2col, convolution fusion, activation/BatchNorm fusion, selective rematerialisation, FP16 accumulation, and native Flow-to-CUDA compilation. These are **not** implemented by this repository.

## Attribution

Architecture, training procedures and CIFAR-10 benchmark motivation were inspired by Ty Sam's [HLB-CIFAR10](https://github.com/tysam-code/hlb-CIFAR10), building on [David Page's fast ResNet CIFAR-10 experiments](https://myrtle.ai/learn/how-to-train-your-resnet/). Compare also [Keller Jordan's airbench](https://github.com/KellerJordan/cifar10-airbench).

CIFAR-10 dataset is by Alex Krizhevsky, Vinod Nair and Geoffrey Hinton (University of Toronto). Download from the original dataset host; training data is not committed to this repository.
