#!/usr/bin/env bash
# Top-1 / Top-10 MCES for one results directory (CPU, HiGHS solver). SOLVER=GUROBI uses GUROBI instead
# (needs `pip install gurobipy` and a license; same values).
#   bash scripts/compute_mces.sh results/marlin_dreams
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
d=${1:?usage: compute_mces.sh RESULTS_DIR}
p="$d/predictions_merged.jsonl"; [ -f "$p" ] || p="$d/predictions.jsonl"
python -m marlin.compute_mces --predictions "$p" --out "$d/mces.json" --top-k 10 \
  --solver "${SOLVER:-HiGHS}" --workers "${WORKERS:-16}" --threads 1 --time-limit 300
