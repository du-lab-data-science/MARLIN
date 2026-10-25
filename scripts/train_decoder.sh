#!/usr/bin/env bash
# Train the MARLIN block-diffusion decoder (self-supervised on SAFE-GPT; multi-GPU via Lightning DDP).
# Needs weights/frigid/DLM.ckpt (warm start) and the SAFE-GPT corpus in the Hugging Face cache:
#   huggingface-cli download datamol-io/safe-gpt --repo-type dataset --revision 16d0be9ad6177ae683a32a86204530e8ee624a0f
# The released checkpoint is step 1,930,000 of this run (global batch 256).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m marlin.train --config-name train_marlin_decoder "$@"
