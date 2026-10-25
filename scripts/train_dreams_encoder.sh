#!/usr/bin/env bash
# Fine-tune the DreaMS backbone and fingerprint head, then embed NPLIB1 (DreaMS environment). Needs:
#   * the official DreaMS weights:            bash scripts/download_weights.sh --dreams
#   * NPLIB1:                                 bash scripts/download_data.sh
#   * optional ICEBERG-simulated spectra (MGF + labels table) from marlin.prep_iceberg_aug ->
#     ms-pred predict_smis.py --sparse-out -> marlin.iceberg_h5_to_mgf; set AUG_MGF / AUG_LABELS
# Training is float32 on purpose: the embedding cache is float32 and a bf16-trained head does not transfer.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
OUT=${OUT:-train_results/dreams_fp}
AUG=()
if [ -n "${AUG_MGF:-}" ]; then
  AUG=(--synthetic-mgf "$AUG_MGF" --aug-labels "${AUG_LABELS:-$AUG_MGF.labels.tsv}" --frac-orig 0.3)
fi
python -m marlin.finetune_dreams_fp_backbone --out-dir "$OUT" --emb-kind backbone \
  --iterative growing --loss-fn cosine --num-splits 4 --iterative-loss-weight 0.4 "${AUG[@]}" \
  --unfreeze-last-k -1 --lr-head 1e-3 --lr-backbone 3e-5 \
  --hidden-dim 1536 --depth 3 --dropout 0.4 --weight-decay 1e-3 \
  --epochs 250 --patience 40 --batch-size 16 --grad-accum 2 --seed 20
python -m marlin.dreams_precompute --spec-folder data/canopus/spec_files --emb-kind backbone \
  --finetuned-backbone "$OUT/backbone_finetuned_fp.pt" --out "$OUT/dreams_embeddings_nplib1.npz"
