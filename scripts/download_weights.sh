#!/usr/bin/env bash
# Fetch every checkpoint / data artifact MARLIN needs and verify checksums.
#
#   bash scripts/download_weights.sh            # everything MARLIN inference and training need
#   bash scripts/download_weights.sh --dreams   # also the official DreaMS weights (only to re-embed spectra)
set -euo pipefail
cd "$(dirname "$0")/.."

# MARLIN weights and data: https://doi.org/10.5281/zenodo.23632849
MARLIN_RECORD="${MARLIN_RECORD:-23632849}"
MARLIN_URL="https://zenodo.org/api/records/$MARLIN_RECORD/files"
FRIGID_TAR_URL="https://zenodo.org/api/records/19685145/files/frigid_pretrained_checkpoints.tar.gz/content"
DREAMS_URL="https://zenodo.org/api/records/10997887/files"

WANT_DREAMS=0
for a in "$@"; do
  case "$a" in
    --dreams) WANT_DREAMS=1 ;;
    *) echo "unknown option $a"; exit 1 ;;
  esac
done

fetch() {  # fetch URL DEST  (skips files that already exist)
  [ -s "$2" ] && return 0
  mkdir -p "$(dirname "$2")"
  wget -O "$2.part" "$1" && mv "$2.part" "$2"
}

# MARLIN weights and data, one archive unpacked at the repository root:
# weights/marlin_decoder.ckpt, weights/dreams_fp/{backbone_finetuned_fp.pt,head.pt},
# weights/mistcf/mistcf_nplib1.ckpt, data/marlin/dreams_embeddings_nplib1.npz, data/marlin/subformulae_mistcf_top1/
MARLIN_TAR=marlin_weights_and_data.tar.gz
if [ ! -s weights/marlin_decoder.ckpt ] || [ ! -s weights/dreams_fp/backbone_finetuned_fp.pt ] \
   || [ ! -s weights/dreams_fp/head.pt ] || [ ! -s weights/mistcf/mistcf_nplib1.ckpt ] \
   || [ ! -s data/marlin/dreams_embeddings_nplib1.npz ] || [ ! -d data/marlin/subformulae_mistcf_top1 ]; then
  fetch "$MARLIN_URL/$MARLIN_TAR/content" "weights/.$MARLIN_TAR"
  tar -xzf "weights/.$MARLIN_TAR"
  rm "weights/.$MARLIN_TAR"
fi

# FRIGID release (CC BY-NC 4.0): the MIST encoder and the decoder checkpoint used for the warm start
if [ ! -s weights/frigid/DLM.ckpt ] || [ ! -s weights/frigid/mist_canopus.pt ]; then
  tmp=$(mktemp -d weights/.frigid_XXXX)
  fetch "$FRIGID_TAR_URL" "$tmp/frigid_pretrained_checkpoints.tar.gz"
  tar -xzf "$tmp/frigid_pretrained_checkpoints.tar.gz" -C "$tmp"
  mkdir -p weights/frigid
  for f in DLM.ckpt mist_canopus.pt; do
    src=$(find "$tmp" -name "$f" -type f | head -n 1)
    [ -n "$src" ] || { echo "ERROR: $f not found in the FRIGID archive"; exit 1; }
    mv "$src" "weights/frigid/$f"
  done
  rm -rf "$tmp"
fi

# official DreaMS weights (CC BY 4.0), only needed to embed new spectra
if [ "$WANT_DREAMS" = 1 ]; then
  fetch "$DREAMS_URL/ssl_model.ckpt/content"       weights/dreams/ssl_model.ckpt
  fetch "$DREAMS_URL/embedding_model.ckpt/content" weights/dreams/embedding_model.ckpt
fi

# SAFE tokenizer (datamol-io/safe-gpt, pinned revision)
HF_HUB_OFFLINE=0 python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("datamol-io/safe-gpt", revision="3d5fa0988383e898d5ac5db7cd52bf715bc37061",
                  allow_patterns=["*.json", "*.txt"])
print("SAFE tokenizer cached")
PY

# verify
sha256sum -c --ignore-missing weights/SHA256SUMS
