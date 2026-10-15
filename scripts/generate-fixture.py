#!/usr/bin/env python3
"""Generate tiny CIFAR-10 binary-format fixtures for offline CI integration tests.

This is NOT the CIFAR-10 dataset and is not used to claim classification accuracy.
It exercises file I/O, normalization, whitening, forward/backward, optimiser and
evaluation without downloading hundreds of megabytes in CI.
"""
from pathlib import Path
from random import Random

root = Path("data/cifar-10-batches-bin")
root.mkdir(parents=True, exist_ok=True)
rng = Random(20261015)

def write(path: Path, count: int) -> None:
    with path.open("wb") as handle:
        for sample in range(count):
            label = sample % 10
            out = bytearray([label])
            for channel in range(3):
                for y in range(32):
                    for x in range(32):
                        pattern = (35 * label + 13 * channel + x * (channel + 1)
                                   + y * (channel + 2) + rng.randint(0, 9)) % 256
                        out.append(pattern)
            assert len(out) == 3073
            handle.write(out)

write(root / "data_batch_1.bin", 16)
write(root / "test_batch.bin", 8)
print("Generated 16 synthetic training and 8 synthetic validation records")
