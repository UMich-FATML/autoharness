#!/bin/bash
set -u

mkdir -p /logs/verifier
echo 0.0 > /logs/verifier/reward.txt

if python3 /tests/test_state.py; then
  exit 0
else
  echo 0.0 > /logs/verifier/reward.txt
  exit 1
fi
