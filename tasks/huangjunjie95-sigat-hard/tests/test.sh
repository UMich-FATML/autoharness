#!/bin/bash
set -euo pipefail

# CANARY: harbor-pwc-sdgnn-hard

score="0.0"
cleanup() {
  printf '%s\n' "$score" > /logs/verifier/reward.txt
}
trap cleanup EXIT

python /app/run_experiment.py \
  --split hidden_test \
  --hidden-module /tests/hidden_test_edges.py \
  --output /app/results.json

python /tests/test_state.py --results /app/results.json

score=$(python - <<'PY'
import json

BASELINE = 0.6735294112296648
TARGET = 0.6946736678888698

with open("/app/results.json", "r", encoding="utf-8") as handle:
    macro_f1 = json.load(handle)["macro_f1"]

reward = (macro_f1 - BASELINE) / (TARGET - BASELINE)
reward = max(0.0, min(1.0, reward))
print(f"{reward:.6f}")
PY
)
