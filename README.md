# HLB-CIFAR10 in Flow

**Native Flow reimplementation of a CIFAR-10 convolutional classifier and training pipeline, designed as a correctness-first baseline for a future GPU training compiler benchmark.**

Upstream research reference: [tysam-code/hlb-CIFAR10](https://github.com/tysam-code/hlb-CIFAR10) (Apache-2.0). Language: [Floooooooooooow/flow](https://github.com/flooooooooooow/flow). This repository was independently written in Flow; no PyTorch or NumPy dependency, and no external C neural-network implementation.

> **Status:** Experimental CPU reference implementation; not an A100 speed record. Numerical test coverage is provided in `smoke.flow`. The GitHub Actions workflow is the authoritative compiler/build gate.

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
| Verification | Finite-difference gradients, pooling adjoint, whitening symmetry, whole-network training smoke |

The native Flow code is in [src/core.flow](src/core.flow), [src/data.flow](src/data.flow) and [src/whitening.flow](src/whitening.flow). All forward passes and gradient updates are implemented in those files.

## Run

Install Flow 2.0 or build its [self-hosted compiler](https://github.com/flooooooooooow/flow). A C11 toolchain is required. Linux and macOS are supported by Flow's stable portable-C backend.

```bash
bash scripts/download-data.sh
flow run smoke.flow
flow run train.flow
```

The default run deliberately uses 256 training and 256 test images, width 8, batch size 8 and 2 epochs. **This is a functional CPU development configuration, not a meaningful CIFAR-10 accuracy benchmark.** The program prints loss, accuracy, EMA accuracy and full CPU training-plus-evaluation time.

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

This is a ground-up implementation of the network and training mechanisms, **not a bit-for-bit reproduction of the PyTorch program**. All floating-point computation is currently F32 on CPU, rather than FP16 channels-last using NVIDIA tensor cores. It computes whitening covariance using four representative 2×2 patches per training image rather than all image patches. It uses a conventional EMA evaluation model rather than upstream's additional EMA parameter write-back. The training schedule uses an integer number of epochs, and its learning-rate update uses equivalent mean-loss scaling rather than the upstream summed-loss convention. First-convolution Dirac initialisation does not reproduce the upstream weight-statistic renormalisation.

Consequently neither the published **94% in under 6.3 seconds** nor **95.79% in roughly 110 seconds** is claimed or expected here. Those were historical upstream measurements on an A100. This repository needs matched accuracy, CUDA kernels, FP16 numerical testing, proper batching optimisation and reproducible end-to-end timing before comparing speed.

## Benchmark protocol

First verify gradient agreement on synthetic small problems with finite differences, then convergence using complete training and held-out validation. Record compiler commit, backend, device, training-data count, batch size, network width, precision, epoch count, random seed, CutMix policy and warm/cold data loading. Keep CIFAR-10's official test split out of hyperparameter selection in serious experiments.

Report data preparation, PCA whitening, training and evaluation times **separately**. The original benchmark deliberately does not include all of them in its headline training time.

CI additionally generates a **synthetic** CIFAR-binary fixture with `scripts/generate-fixture.py` and performs an end-to-end train/evaluate run. Those generated pixels are not part of the official CIFAR-10 dataset and must never be used for accuracy claims.

The immediate performance roadmap is batched GEMM/im2col, convolution fusion, activation/BatchNorm fusion, selective rematerialisation, FP16 accumulation, and native Flow-to-CUDA compilation. These are **not** implemented by this repository.

## Attribution

Architecture, training procedures and CIFAR-10 benchmark motivation were inspired by Ty Sam's [HLB-CIFAR10](https://github.com/tysam-code/hlb-CIFAR10), building on [David Page's fast ResNet CIFAR-10 experiments](https://myrtle.ai/learn/how-to-train-your-resnet/). Compare also [Keller Jordan's airbench](https://github.com/KellerJordan/cifar10-airbench).

CIFAR-10 dataset is by Alex Krizhevsky, Vinod Nair and Geoffrey Hinton (University of Toronto). Download from the original dataset host; training data is not committed to this repository.
