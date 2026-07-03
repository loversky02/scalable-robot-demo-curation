#!/usr/bin/env bash
# Offline verification: unit tests + synthetic ablation. No network, no GPU, $0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== unit tests =="
python3 -c "import pytest,sys; sys.exit(pytest.main(['-q','tests']))"

echo
echo "== M1 synthetic ablation (writes outputs/) =="
python3 experiments/run_ablation.py --source synthetic

echo
echo "== M2.5 reward-free proxy study (writes outputs/) =="
python3 experiments/proxy_vs_oracle.py

echo
echo "OK — offline verification complete."
