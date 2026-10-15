#!/usr/bin/env python3
"""Convert real, independently sourced datasets to the Flow CIFAR-record ABI.

This is a preprocessing-only adapter, not a learned Python baseline. Every
record is [label u8][3072 channel-first uint8 values]. Feature encodings
are explicitly modality-specific; using the image CNN does not by itself
establish modality-appropriate accuracy parity.

Datasets:
  fashion-mnist: ten classes, 28x28 grey garments
  spoken-digits: ten classes, waveform-derived log-magnitude spectrograms
  uci-har: six classes, nine-axis inertial time series, subject-disjoint
  ag-news: four classes, byte-coded news headlines and article text
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import gzip
import io
import json
from pathlib import Path
import struct
from urllib.request import Request, urlopen
import wave
import zipfile

import numpy as np

FASHION_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
AUDIO_URL = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip"
HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
NEWS_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/"


def remote(url: str) -> bytes:
    with urlopen(Request(url, headers={"User-Agent": "hlb-cifar-flow/1.0"}), timeout=90) as response:
        return response.read()


def payload(image: np.ndarray, label: int) -> bytes:
    if image.shape != (3, 32, 32):
        raise ValueError(f"Invalid encoded feature shape: {image.shape}")
    if not 0 <= label < 10:
        raise ValueError(f"Class index {label} is outside Flow's ten-logit head")
    return bytes([label]) + np.asarray(image, dtype=np.uint8).tobytes()


def write_split(root: Path, name: str, rows, cap: int, classes: int):
    # Balancing is needed because upstream IDX/CSV files are class-sorted.
    counts: Counter[int] = Counter()
    collected: list[tuple[int, np.ndarray]] = []
    if cap < classes:
        raise ValueError("Need at least one example per class")
    quotas = {label: cap // classes + (1 if label < cap % classes else 0)
              for label in range(classes)}
    for label, image in rows:
        if counts[label] >= quotas[label]:
            continue
        collected.append((label, image))
        counts[label] += 1
        if all(counts[i] >= quotas[i] for i in range(classes)):
            break
    if any(counts[i] == 0 for i in range(classes)):
        raise RuntimeError(f"{name}: source sample omitted at least one class: {dict(counts)}")
    rng = np.random.default_rng(20261015)
    rng.shuffle(collected)
    if len(collected) != cap:
        raise RuntimeError(f"{name}: expected {cap}, received {len(collected)}")
    collected = collected[:cap]
    output = root / ("data_batch_1.bin" if name == "train" else "test_batch.bin")
    with output.open("wb") as handle:
        for label, image in collected:
            handle.write(payload(image, label))
    return {"count": len(collected), "per_class": dict(Counter(label for label, _ in collected)),
            "sha256": sha256_file(output), "filename": output.name}


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def grayscale_32(image: np.ndarray):
    out = np.zeros((32, 32), dtype=np.uint8)
    out[2:30, 2:30] = image.reshape(28, 28)
    return np.broadcast_to(out, (3, 32, 32)).copy()


def load_fashion():
    base = FASHION_URL
    def idx(split):
        imagefile = "train-images-idx3-ubyte.gz" if split == "train" else "t10k-images-idx3-ubyte.gz"
        labelfile = "train-labels-idx1-ubyte.gz" if split == "train" else "t10k-labels-idx1-ubyte.gz"
        images = gzip.decompress(remote(base + imagefile))
        labels = gzip.decompress(remote(base + labelfile))
        magic, n, h, w = struct.unpack(">4I", images[:16])
        label_magic, label_n = struct.unpack(">2I", labels[:8])
        if (magic, label_magic, n, h, w, label_n) != (2051, 2049, n, 28, 28, n):
            raise ValueError("Invalid Fashion-MNIST IDX headers")
        raw = np.frombuffer(images, dtype=np.uint8, offset=16).reshape(n, 28, 28)
        cls = np.frombuffer(labels, dtype=np.uint8, offset=8)
        return ((int(y), grayscale_32(x)) for x, y in zip(raw, cls))
    return idx("train"), idx("test"), 10, {"source": "zalandoresearch/fashion-mnist"}


def log_spectrogram(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
        channels, rate, width = wav.getnchannels(), wav.getframerate(), wav.getsampwidth()
        if width not in (1, 2) or channels < 1:
            raise ValueError("Only PCM u8/i16 audio supported")
        raw = wav.readframes(wav.getnframes())
    dtype = np.int16 if width == 2 else np.uint8
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    audio = audio.reshape(-1, channels).mean(axis=1)
    if width == 1:
        audio -= 128.0
    audio /= 32768.0 if width == 2 else 128.0
    if audio.size < 32:
        raise ValueError("Audio recording too short")
    # Time-resample to 8192 samples (also handles sampling-rate variations).
    dest = np.linspace(0, audio.size - 1, 8192)
    audio = np.interp(dest, np.arange(audio.size), audio).astype(np.float32)
    window = np.hanning(512).astype(np.float32)
    spectrogram = []
    for start in np.linspace(0, 8192 - 512, 32, dtype=np.int32):
        spectrum = np.abs(np.fft.rfft(audio[start:start + 512] * window))
        # Linear interpolation across 0..4000 Hz to 32 frequency bins.
        freq = np.linspace(0, min(4000, rate / 2), 32)
        bins = np.fft.rfftfreq(512, 1 / rate)
        spectrogram.append(np.interp(freq, bins, spectrum))
    spec = np.log1p(np.asarray(spectrogram, dtype=np.float32).T)
    # Explicit fixed clip scale instead of fitting normalisation on test data.
    spec = (np.clip(spec / 5.0, 0.0, 1.0) * 255).astype(np.uint8)
    return np.stack((spec, spec, spec))


def load_spoken_digits():
    archive = zipfile.ZipFile(io.BytesIO(remote(AUDIO_URL)))
    names = sorted(n for n in archive.namelist()
                   if "/recordings/" in n and n.lower().endswith(".wav"))
    def generate(train):
        for name in names:
            base = Path(name).name
            parts = base.split("_")
            if len(parts) != 3:
                continue
            label, _, index = parts
            if not label.isdigit():
                continue
            # Official FSDD split: recording indices 0..4 held out for testing.
            if (int(index.removesuffix(".wav")) >= 5) != train:
                continue
            yield int(label), log_spectrogram(archive.read(name))
    return generate(True), generate(False), 10, {
        "source": "Jakobovski/free-spoken-digit-dataset", "split": "official index 0..4 test, 5..49 train",
        "license": "CC-BY-SA-4.0", "encoding": "log magnitude STFT"}


def uci_har_image(samples: np.ndarray):
    # Inertial inputs are 9 independent 128-frame time series.
    if samples.shape != (9, 128):
        raise ValueError(f"Expected 9x128 UCI HAR signal, got {samples.shape}")
    channels = []
    for group in range(3):
        signals = samples[3 * group:3 * group + 3].reshape(12, 32)
        grid = np.stack([np.interp(np.linspace(0, 11, 32),
                                   np.arange(12), signals[:, j]) for j in range(32)], axis=1)
        # Fixed -4 .. +4 sensor-unit mapping, shared across splits.
        channels.append((np.clip((grid + 4.0) / 8.0, 0, 1) * 255).astype(np.uint8))
    return np.stack(channels)


def load_uci_har():
    archive = zipfile.ZipFile(io.BytesIO(remote(HAR_URL)))
    prefix = next((item for item in archive.namelist() if item.endswith("UCI HAR Dataset/train/y_train.txt")), None)
    if not prefix:
        raise RuntimeError("UCI HAR archive layout changed")
    root = prefix.removesuffix("train/y_train.txt")
    axis_names = ("body_acc_x", "body_acc_y", "body_acc_z",
                  "body_gyro_x", "body_gyro_y", "body_gyro_z",
                  "total_acc_x", "total_acc_y", "total_acc_z")
    def generate(split):
        y = np.loadtxt(io.BytesIO(archive.read(root + f"{split}/y_{split}.txt")), dtype=np.int32) - 1
        signals = []
        for axis in axis_names:
            name = root + f"{split}/Inertial Signals/{axis}_{split}.txt"
            signals.append(np.loadtxt(io.BytesIO(archive.read(name)), dtype=np.float32))
        x = np.stack(signals, axis=1)
        for i in range(len(y)):
            yield int(y[i]), uci_har_image(x[i])
    return generate("train"), generate("test"), 6, {
        "source": "UCI HAR 240", "split": "official subject-disjoint train/test",
        "license": "CC-BY-4.0", "encoding": "3 sensor families x 32x32"}


def text_features(title: str, description: str):
    # This is a character-level baseline, NOT a pretrained text encoder.
    # Byte grids preserve local ordering but impose an image-like topology.
    src = (title + "\n" + description).encode("utf-8")[:3072]
    padded = np.zeros(3072, dtype=np.uint8)
    padded[:len(src)] = np.frombuffer(src, dtype=np.uint8)
    return padded.reshape(3, 32, 32)


def load_ag_news():
    def generate(split):
        # Iterate a stream rather than retaining all 120k train examples.
        with urlopen(Request(NEWS_URL + f"{split}.csv",
                             headers={"User-Agent": "hlb-cifar-flow/1.0"}), timeout=120) as response:
            stream = io.TextIOWrapper(response, encoding="utf-8")
            for row in csv.reader(stream):
                if len(row) < 3:
                    continue
                yield int(row[0]) - 1, text_features(row[1], row[2])
    return generate("train"), generate("test"), 4, {
        "source": "AG News CharCnn_Keras CSV", "encoding": "first 3072 UTF-8 bytes in 3x32x32"}


LOADERS = {"fashion-mnist": load_fashion, "spoken-digits": load_spoken_digits,
           "uci-har": load_uci_har, "ag-news": load_ag_news}


def run(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    train, test, classes, provenance = LOADERS[args.dataset]()
    training = write_split(root, "train", train, args.train, classes)
    testing = write_split(root, "test", test, args.test, classes)
    metadata = {"name": args.dataset, "modality": {
        "fashion-mnist": "vision", "spoken-digits": "audio",
        "uci-har": "inertial_time_series", "ag-news": "text"}[args.dataset],
        "classes": classes, "feature_shape": [3, 32, 32],
        "train": training, "test": testing, **provenance}
    (root / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(LOADERS), required=True)
    parser.add_argument("--train", type=int, default=64)
    parser.add_argument("--test", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("data/cifar-10-batches-bin"))
    args = parser.parse_args()
    if args.train < 20 or args.test < 10:
        parser.error("at least 20 train and 10 test samples required for stratified tiny experiments")
    run(args)


if __name__ == "__main__":
    main()
