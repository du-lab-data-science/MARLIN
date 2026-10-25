#!/usr/bin/env bash
# Reproduce the MARLIN results on the NPLIB1 test split.
#
#   bash scripts/reproduce_paper.sh main        # MARLIN with the DreaMS and the MIST encoder
#   bash scripts/reproduce_paper.sh ablations   # inference-time ablations (DreaMS encoder)
#   NUM_SHARDS=4 SHARD_IDX=0 bash scripts/reproduce_paper.sh main   # strided shard i of N (one GPU each)
#
# Each run writes results/<name>/predictions.jsonl (+ metrics). With shards, merge first:
#   python -m marlin.merge_shards results/<name>
# Then compute MCES (scripts/compute_mces.sh results/<name>) and print the metrics:
#   python -m marlin.summarize results/<name>
# Runtime: about 34 s per spectrum on one NVIDIA L40S (384 decodes per spectrum).
# Reference outputs of these runs are in paper_results/.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
NUM_SHARDS=${NUM_SHARDS:-1}; SHARD_IDX=${SHARD_IDX:-0}
SHARD="--num-shards $NUM_SHARDS --shard-idx $SHARD_IDX"
DREAMS=src/marlin/configs/infer_nplib1_dreams.yaml
MIST=src/marlin/configs/infer_nplib1_mist.yaml

marlin() {   # marlin NAME CONFIG [extra infer flags...]
  local name=$1 cfg=$2; shift 2
  python -m marlin.infer --config "$cfg" --output-dir "results/$name" $SHARD "$@"
}

case "${1:-}" in
  main)
    marlin marlin_dreams "$DREAMS"
    marlin marlin_mist "$MIST"
    ;;
  ablations)
    marlin abl_dreams_wo_conditioning_diversity "$DREAMS" --fp-diversity 0.0
    marlin abl_dreams_wo_mass_conditioning "$DREAMS" --no-mass-conditioning --no-mass-shell-prune
    marlin abl_dreams_wo_prune "$DREAMS" --no-mass-shell-prune
    ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
