#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl --fail --location --retry 3 \
  "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz" \
  --output "data/cifar-10-binary.tar.gz"
tar -xzf "data/cifar-10-binary.tar.gz" -C data
test -s "data/cifar-10-batches-bin/data_batch_1.bin"
test -s "data/cifar-10-batches-bin/test_batch.bin"
echo "CIFAR-10 binary dataset ready in data/cifar-10-batches-bin"
