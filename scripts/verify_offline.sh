#!/usr/bin/env bash
# Offline verification: unit tests + synthetic ablation. No network, no GPU, $0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== unit tests =="
python3 -c "import pytest,sys; sys.exit(pytest.main(['-q','tests']))"

echo
echo "== synthetic ablation (writes outputs/) =="
python3 experiments/run_ablation.py --source synthetic

echo
echo "OK — offline verification complete."
