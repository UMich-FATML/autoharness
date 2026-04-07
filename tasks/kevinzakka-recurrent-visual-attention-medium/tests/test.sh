#!/bin/bash
set -euo pipefail

# CANARY: harbor-pwc-rva-medium

score="0.0"
mkdir -p /logs/verifier
cleanup() {
  printf '%s\n' "$score" > /logs/verifier/reward.txt
}
trap cleanup EXIT

python /app/main.py \
  --split hidden_test \
  --hidden-module /tests/hidden_split.py \
  --output /app/results.json

python /tests/test_state.py --results /app/results.json

score=$(python - <<'PY'
import json

BASELINE = 0.1
TARGET = 0.9458333333333333

with open("/app/results.json", "r", encoding="utf-8") as handle:
    accuracy = json.load(handle)["accuracy"]

reward = (accuracy - BASELINE) / (TARGET - BASELINE)
reward = max(0.0, min(1.0, reward))
print(f"{reward:.6f}")
PY
)
