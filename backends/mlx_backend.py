#!/usr/bin/env python3
"""Apple MLX backend for the Flow HLB CNN.

MLX performs convolutions, reductions, differentiation and optimisation on
Apple Silicon. Flow's version-2 binary checkpoint is the common interchange
format. No PyTorch import or neural-network shim is used.

Modes:
  parity:  compare native Flow's parity.flow logits to MLX on identical weights
  infer:   evaluate a saved Flow model on CIFAR-format binary records
  train:   optimise imported Flow weights with MLX autograd and save a v2 model
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import time
import numpy as np

import mlx.core as mx

MAGIC = 0x484C4246
NCLASSES = 10
SQRT2 = math.sqrt(2.0)
BLOCKS = ("first", "second", "third")


def shapes(width: int) -> dict[str, tuple[int, ...]]:
    sizes: dict[str, tuple[int, ...]] = {"whitening": (24, 3, 2, 2)}
    cin = 24
    for name, cout in zip(BLOCKS, (width, 4 * width, 8 * width)):
        sizes[f"{name}.a"] = (cout, cin, 3, 3)
        sizes[f"{name}.b"] = (cout, cout, 3, 3)
        for side in ("norm_a", "norm_b"):
            for field in ("bias", "running_mean", "running_var"):
                sizes[f"{name}.{side}.{field}"] = (cout,)
        cin = cout
    sizes["head"] = (width * 8, NCLASSES)
    return sizes


def read_checkpoint(path: Path) -> tuple[int, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        header = struct.unpack("<8i", handle.read(32))
        if header[0] != MAGIC or header[1] != 2:
            raise ValueError(f"{path}: requires Flow checkpoint format v2")
        width = header[2]
        if not 2 <= width <= 64 or header[3:] != (24, width, 4 * width, 8 * width, NCLASSES):
            raise ValueError(f"{path}: invalid checkpoint geometry")
        def read_floats(count: int) -> np.ndarray:
            raw = handle.read(4 * count)
            if len(raw) != 4 * count:
                raise ValueError(f"{path}: truncated checkpoint")
            return np.frombuffer(raw, dtype="<f4").copy()
        mean = read_floats(3)
        invstd = read_floats(3)
        arrays = {name: read_floats(math.prod(shape)).reshape(shape)
                  for name, shape in shapes(width).items()}
        if handle.read(1):
            raise ValueError(f"{path}: trailing bytes")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(invstd)):
        raise ValueError("Non-finite preprocessing values")
    return width, arrays, mean, invstd


def write_checkpoint(path: Path, width: int, params: dict[str, mx.array],
                     running: dict[str, mx.array], mean: np.ndarray,
                     invstd: np.ndarray) -> None:
    tensor_shapes = shapes(width)
    with path.open("wb") as handle:
        handle.write(struct.pack("<8i", MAGIC, 2, width, 24, width, width * 4, width * 8, NCLASSES))
        for stat in (mean, invstd):
            handle.write(np.asarray(stat, dtype="<f4").tobytes())
        for name, shape in tensor_shapes.items():
            src = running[name] if name in running else params[name]
            arr = np.asarray(src, dtype="<f4")
            expected = (shape[0], shape[2], shape[3], shape[1]) if len(shape) == 4 else shape
            if arr.shape != expected or not np.all(np.isfinite(arr)):
                raise ValueError(f"{name}: invalid tensor shape or non-finite data")
            if len(shape) == 4:
                arr = arr.transpose(0, 3, 1, 2)  # MLX O,K,K,I -> Flow O,I,K,K
            handle.write(np.ascontiguousarray(arr).tobytes())


def load_parameters(arrays: dict[str, np.ndarray]):
    params, running = {}, {}
    for name, a in arrays.items():
        item = mx.array(a.transpose(0, 2, 3, 1) if a.ndim == 4 else a)
        if ".running_" in name:
            running[name] = item
        else:
            params[name] = item
    mx.eval(params, running)
    return params, running


def gelu(x):
    return x * 0.5 * (1.0 + mx.erf(x / SQRT2))


def conv(params, name, x, pad):
    return mx.conv2d(x, params[name], padding=pad)


def max_pool2(x):
    n, h, w, c = x.shape
    # Flow's max-pool floors odd spatial extents (31 -> 15), matching
    # PyTorch MaxPool2d(2). Dropping the last row/column is intentional.
    x = x[:, : (h // 2) * 2, : (w // 2) * 2, :]
    return mx.max(mx.reshape(x, (n, h // 2, 2, w // 2, 2, c)), axis=(2, 4))


def norm(x, name, params, running, training, moments):
    if training:
        mean = mx.mean(x, axis=(0, 1, 2))
        delta = x - mean
        var = mx.mean(delta * delta, axis=(0, 1, 2))
        moments.append((name, mean, var, int(x.shape[0] * x.shape[1] * x.shape[2])))
    else:
        mean = running[f"{name}.running_mean"]
        var = running[f"{name}.running_var"]
    return (x - mean) / mx.sqrt(var + 1.0e-12) + params[f"{name}.bias"]


def forward(params, running, x, training=False):
    """NHWC input, ten logits; the same geometry and /9 head as Flow."""
    moments = []
    x = gelu(conv(params, "whitening", x, 0))
    for name in BLOCKS:
        x = conv(params, f"{name}.a", x, 1)
        x = max_pool2(x)
        x = gelu(norm(x, f"{name}.norm_a", params, running, training, moments))
        x = conv(params, f"{name}.b", x, 1)
        x = gelu(norm(x, f"{name}.norm_b", params, running, training, moments))
    logits = mx.matmul(mx.max(x, axis=(1, 2)), params["head"]) / 9.0
    return logits, moments


def parity_inputs() -> np.ndarray:
    # Exactly the f32 arithmetic from parity.flow, in NCHW.
    values = (np.arange(2 * 3 * 32 * 32, dtype=np.int64) * 13) % 251
    return (values.astype(np.float32) / np.float32(251.0)).reshape(2, 3, 32, 32)


def _as_nhwc(x: np.ndarray):
    return mx.array(np.ascontiguousarray(x.transpose(0, 2, 3, 1)), dtype=mx.float32)


def run_parity(args):
    width, arrays, _, _ = read_checkpoint(Path(args.checkpoint))
    params, running = load_parameters(arrays)
    logits, _ = forward(params, running, _as_nhwc(parity_inputs()))
    mx.eval(logits)
    candidate = np.asarray(logits, dtype=np.float32)
    lines = Path(args.flow_logits).read_text().splitlines()
    start, stop = lines.index("PARITY_START"), lines.index("PARITY_END")
    baseline = np.array([float(line.strip()) for line in lines[start + 1:stop]], dtype=np.float32)
    if baseline.size != 2 * NCLASSES:
        raise ValueError("Expected 20 Flow logits, found " + str(baseline.size))
    baseline = baseline.reshape(2, NCLASSES)
    abs_delta = np.abs(candidate - baseline)
    tolerance = args.atol + args.rtol * np.abs(baseline)
    passed = bool(np.all(abs_delta <= tolerance))
    summary = {"backend": "mlx", "test": "flow_checkpoint_logit_parity",
               "width": width, "max_abs_error": float(abs_delta.max()),
               "max_rel_error": float((abs_delta / np.maximum(1e-6, np.abs(baseline))).max()),
               "atol": args.atol, "rtol": args.rtol, "passed": passed}
    print(json.dumps(summary, sort_keys=True))
    if not passed:
        raise SystemExit(1)



def run_grad_parity(args):
    width, arrays, _, _ = read_checkpoint(Path(args.checkpoint))
    params, running = load_parameters(arrays)
    frozen = params.pop("whitening")
    x = _as_nhwc(parity_inputs())
    y = mx.array(np.eye(NCLASSES, dtype=np.float32)[[0, 1]])

    def loss_fn(trainable):
        weights = dict(trainable)
        weights["whitening"] = frozen
        logits, _ = forward(weights, running, x, training=True)
        target = 0.8 * y + 0.02
        return -mx.mean(mx.sum(target * (logits - mx.logsumexp(
            logits, axis=-1, keepdims=True)), axis=-1))

    loss, gradients = mx.value_and_grad(loss_fn)(params)
    mx.eval(loss, gradients)
    def weight_gradient(name):
        values = np.asarray(gradients[name], dtype=np.float32)
        if values.ndim == 4:
            values = values.transpose(0, 3, 1, 2)
        return values.reshape(-1)

    measured = np.array([
        float(np.asarray(loss)),
        weight_gradient("head")[0], weight_gradient("head")[1],
        weight_gradient("head")[10],
        weight_gradient("first.a")[0], weight_gradient("first.a")[1],
        weight_gradient("first.a")[10],
        weight_gradient("second.b")[0], weight_gradient("second.b")[4],
        weight_gradient("third.norm_a.bias")[0],
        weight_gradient("third.norm_b.bias")[0],
        weight_gradient("first.norm_a.bias")[0]
    ], dtype=np.float32)
    lines = Path(args.flow_gradients).read_text().splitlines()
    start, stop = lines.index("GRAD_START"), lines.index("GRAD_END")
    baseline = np.array([float(line.strip()) for line in lines[start + 1:stop]], dtype=np.float32)
    if measured.shape != baseline.shape or len(baseline) != 12:
        raise ValueError(f"Expected 12 Flow loss/gradient values, got {len(baseline)}")
    diffs = np.abs(measured - baseline)
    passed = bool(np.all(diffs <= args.grad_atol + args.grad_rtol * np.abs(baseline)))
    print(json.dumps({"backend": "mlx", "test": "flow_analytic_gradient_parity",
                      "count": len(measured), "max_abs_error": float(diffs.max()),
                      "passed": passed, "atol": args.grad_atol, "rtol": args.grad_rtol}))
    if not passed:
        print("flow  :", baseline.tolist())
        print("mlx   :", measured.tolist())
        raise SystemExit(1)

def read_binary_records(folder: Path, limit: int, train: bool):
    paths = ([folder / f"data_batch_{i}.bin" for i in range(1, 6)] if train
             else [folder / "test_batch.bin"])
    records = []
    for path in paths:
        if not path.exists():
            if records:
                break
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        if len(raw) % 3073:
            raise ValueError(f"{path}: non-integral CIFAR record count")
        records.append(np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3073))
        if sum(len(x) for x in records) >= limit:
            break
    block = np.concatenate(records)[:limit]
    labels = block[:, 0].astype(np.int32)
    if len(labels) == 0 or labels.max() >= NCLASSES:
        raise ValueError("Expected 1..10 class CIFAR-format records with labels 0..9")
    x = block[:, 1:].astype(np.float32).reshape(-1, 3, 32, 32) / 255.0
    return x, labels


def normalized(x, mean, invstd):
    return (x - mean[None, :, None, None]) * invstd[None, :, None, None]


def evaluate(params, running, x, labels, batch, return_predictions=False):
    correct = 0
    predictions_all = []
    for start in range(0, len(labels), batch):
        xb = _as_nhwc(x[start:start + batch])
        out, _ = forward(params, running, xb)
        mirror, _ = forward(params, running, xb[:, :, ::-1, :])
        logits = (out + mirror) * 0.5
        mx.eval(logits)
        guess = np.asarray(mx.argmax(logits, axis=-1))
        correct += int(np.sum(guess == labels[start:start + batch]))
        if return_predictions:
            predictions_all.extend(int(v) for v in guess)
    accuracy = correct / len(labels)
    return (accuracy, predictions_all) if return_predictions else accuracy


def run_infer(args):
    width, arrays, mean, invstd = read_checkpoint(Path(args.checkpoint))
    params, running = load_parameters(arrays)
    x, labels = read_binary_records(Path(args.data_dir), args.limit, train=False)
    x = normalized(x, mean, invstd)
    start = time.perf_counter()
    accuracy, predictions = evaluate(params, running, x, labels, args.batch, True)
    elapsed = time.perf_counter() - start
    print(json.dumps({"backend": "mlx", "mode": "inference", "model_width": width,
                      "examples": len(labels), "accuracy": accuracy,
                      "predictions": predictions, "elapsed_seconds": elapsed}, sort_keys=True))


def run_train(args):
    width, arrays, mean, invstd = read_checkpoint(Path(args.checkpoint))
    params, running = load_parameters(arrays)
    raw_x, labels = read_binary_records(Path(args.data_dir), args.limit, train=True)
    if args.holdout < 1 or args.holdout >= len(labels):
        raise ValueError("holdout must leave both validation and training examples")
    # Prevent leakage: this training path is for continuing a Flow checkpoint.
    # It retains that checkpoint's train-only normalisation statistics.
    x = normalized(raw_x, mean, invstd)
    train_x, valid_x = x[:-args.holdout], x[-args.holdout:]
    train_y, valid_y = labels[:-args.holdout], labels[-args.holdout:]
    velocity = {name: mx.zeros_like(weight) for name, weight in params.items()
                if name != "whitening"}
    frozen = params.pop("whitening")  # frozen data-dependent filters
    params_with_frozen = dict(params)
    params_with_frozen["whitening"] = frozen

    def criterion(trainable, batch, targets):
        all_params = dict(trainable)
        all_params["whitening"] = frozen
        logits, moments = forward(all_params, running, batch, training=True)
        smoothed = 0.8 * targets + 0.02
        log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        loss = -mx.mean(mx.sum(smoothed * log_probs, axis=-1))
        return loss, moments

    grad_fn = mx.value_and_grad(criterion)
    n_steps = args.epochs * math.ceil(len(train_x) / args.batch)
    step = 0
    start = time.perf_counter()
    for epoch in range(args.epochs):
        rng = np.random.default_rng(args.seed + epoch)
        order = rng.permutation(len(train_x))
        last_loss = 0.0
        for offset in range(0, len(train_x), args.batch):
            indices = order[offset:offset + args.batch]
            inp = _as_nhwc(train_x[indices])
            targets = mx.array(np.eye(NCLASSES, dtype=np.float32)[train_y[indices]])
            (loss, moments), grads = grad_fn(params, inp, targets)
            progress = step / max(1, n_steps - 1)
            lr = args.lr_scale * 1.525 * (progress / .23 if progress < .23 else
                          1.0 - .93 * (progress - .23) / .77)
            updated = {}
            for name, weight in params.items():
                bias = name.endswith(".bias")
                rate = lr * (64 if bias else 1)
                decay = 0.0006687 * len(indices) / 512 * (1 / 64 if bias else 1)
                grad = grads[name] + decay * weight
                velocity[name] = 0.85 * velocity[name] + grad
                updated[name] = weight - rate * (grad + .85 * velocity[name])
            for name, batch_mean, batch_var, count in moments:
                update_var = batch_var * (count / (count - 1)) if count > 1 else batch_var
                running[f"{name}.running_mean"] = (
                    .6 * running[f"{name}.running_mean"] + .4 * batch_mean)
                running[f"{name}.running_var"] = (
                    .6 * running[f"{name}.running_var"] + .4 * update_var)
            params = updated
            mx.eval(params, running, velocity, loss)
            last_loss = float(np.asarray(loss))
            step += 1
        all_params = dict(params)
        all_params["whitening"] = frozen
        accuracy = evaluate(all_params, running, valid_x, valid_y, args.batch)
        print(json.dumps({"backend": "mlx", "epoch": epoch + 1,
                          "train_loss": last_loss, "validation_accuracy": accuracy}))
    params["whitening"] = frozen
    write_checkpoint(Path(args.output), width, params, running, mean, invstd)
    print(json.dumps({"backend": "mlx", "mode": "finetune", "epochs": args.epochs,
                      "training_examples": len(train_y), "validation_examples": len(valid_y),
                      "elapsed_seconds": time.perf_counter() - start,
                      "checkpoint": args.output}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("parity", "grad-parity", "infer", "train"))
    parser.add_argument("--checkpoint", default="hlb-ema.bin")
    parser.add_argument("--flow-logits", default="flow-parity.txt")
    parser.add_argument("--flow-gradients", default="flow-grads.txt")
    parser.add_argument("--data-dir", default="data/cifar-10-batches-bin")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--holdout", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--output", default="mlx-finetuned.bin")
    parser.add_argument("--atol", type=float, default=0.004)
    parser.add_argument("--rtol", type=float, default=0.002)
    parser.add_argument("--grad-atol", type=float, default=0.005)
    parser.add_argument("--grad-rtol", type=float, default=0.02)
    args = parser.parse_args()
    {"parity": run_parity, "grad-parity": run_grad_parity, "infer": run_infer, "train": run_train}[args.mode](args)


if __name__ == "__main__":
    main()
