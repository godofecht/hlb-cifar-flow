#!/usr/bin/env python3
"""Build genuine CIFAR-10 mini-splits for the Flow WASM browser experiment."""
import hashlib
import json
import random
import tarfile
import urllib.request
from pathlib import Path

URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
MD5 = "c32a1d4ab5d03f1284b67883e8d87530"
RECORD = 3073
TRAIN_COUNT = 64
TEST_COUNT = 32
SEED = 20261106

def select_records(payload: bytes, count: int, seed: int) -> bytes:
    if len(payload) != RECORD * 10000:
        raise ValueError(f"Unexpected CIFAR-10 batch size: {len(payload)}")
    indices = sorted(random.Random(seed).sample(range(10000), count))
    selected = b"".join(payload[i * RECORD:(i + 1) * RECORD] for i in indices)
    if len(selected) != count * RECORD:
        raise AssertionError("Incomplete selected subset")
    if any(selected[i * RECORD] > 9 for i in range(count)):
        raise ValueError("Invalid CIFAR-10 label in sample")
    return selected

def main() -> None:
    output = Path("site/data")
    output.mkdir(parents=True, exist_ok=True)
    archive = Path("build/cifar-10-binary.tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        print("Downloading official CIFAR-10 binary archive", flush=True)
        urllib.request.urlretrieve(URL, archive)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != MD5:
        raise ValueError(f"CIFAR-10 binary archive checksum mismatch: {digest}")
    with tarfile.open(archive, "r:gz") as tar:
        def read(member: str) -> bytes:
            entry = tar.extractfile("cifar-10-batches-bin/" + member)
            if entry is None:
                raise FileNotFoundError(member)
            return entry.read()
        train = select_records(read("data_batch_1.bin"), TRAIN_COUNT, SEED)
        test = select_records(read("test_batch.bin"), TEST_COUNT, SEED + 1)
    (output / "train.bin").write_bytes(train)
    (output / "test.bin").write_bytes(test)
    meta = {
        "dataset": "CIFAR-10 official binary records",
        "source": URL,
        "source_md5": MD5,
        "seed": SEED,
        "sampling": "Python random.sample without replacement; sorted offsets",
        "training_records": TRAIN_COUNT,
        "test_records": TEST_COUNT,
        "record_size": RECORD,
        "training_split": "official data_batch_1",
        "evaluation_split": "official test_batch",
        "classes": ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"],
        "accuracy_note": "Tiny subsets; not a CIFAR-10 accuracy benchmark"
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Verified original data: {TRAIN_COUNT} train / {TEST_COUNT} test")

if __name__ == "__main__":
    main()
