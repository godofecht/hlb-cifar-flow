#!/usr/bin/env python3
"""Require sample-by-sample classification parity, not merely equal accuracy."""
import argparse
import json
from pathlib import Path
import re

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("flow")
    ap.add_argument("mlx")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", default="parity-result.json")
    args = ap.parse_args()
    native = Path(args.flow).read_text()
    prediction_pairs = re.findall(r"^PRED (\d+) (\d+)\s*$", native, flags=re.M)
    flow_predictions = [int(c) for i, c in sorted(
        ((int(i), int(c)) for i, c in prediction_pairs), key=lambda x: x[0])]
    recorded_indices = [int(i) for i, _ in prediction_pairs]
    if sorted(recorded_indices) != list(range(len(recorded_indices))):
        raise ValueError("Flow prediction indexes missing or duplicated")
    match = re.search(r"^accuracy\s+([0-9.]+)\s*$", native, flags=re.M)
    if not match:
        raise ValueError("No native Flow accuracy reported")
    flow_accuracy = float(match.group(1))
    mlx_lines = Path(args.mlx).read_text().splitlines()
    mlx_record = next((json.loads(line) for line in mlx_lines
                       if line.startswith("{") and '"mode": "inference"' in line), None)
    if mlx_record is None:
        raise ValueError("No MLX inference record found")
    mlx_predictions = mlx_record["predictions"]
    parity = (len(flow_predictions) == mlx_record["examples"] and
              flow_predictions == mlx_predictions and
              abs(flow_accuracy - mlx_record["accuracy"]) <= 1.0e-6)
    result = {"dataset": args.dataset, "test_examples": len(flow_predictions),
              "flow_accuracy": flow_accuracy, "mlx_accuracy": mlx_record["accuracy"],
              "prediction_agreement": sum(a == b for a, b in zip(
                  flow_predictions, mlx_predictions)) / max(1, len(flow_predictions)),
              "exact_classification_parity": parity}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    if not parity:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
