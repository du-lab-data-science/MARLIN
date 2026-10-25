"""Warm-start the MARLIN decoder from FRIGID's released ``DLM.ckpt``.

Backbone and fingerprint-encoder weights are copied where names and shapes match (the model must use
the checkpoint's sizes, hidden size 896); the mass and isotope modules start from random weights.

    python -m marlin.warm_start --dlm-ckpt weights/frigid/DLM.ckpt   # report what transfers
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Tuple

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_PROJECT_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def remap_dlm_state_dict(dlm_sd: Dict[str, torch.Tensor], target_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Translate a FRIGID-base state_dict into MARLIN keys (shape-checked)."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in dlm_sd.items():
        if not hasattr(v, "shape"):
            continue
        if k == "token_atom_counts":
            continue
        if k.startswith("formula_conditioner."):
            continue  # no formula conditioning
        if k.startswith("fingerprint_conditioner.fingerprint_encoder."):
            nk = "conditioner." + k[len("fingerprint_conditioner."):]
        elif k.startswith("fingerprint_conditioner."):
            continue  # FRIGID's separate fingerprint cross-attention is not used
        else:
            nk = k
        if nk in target_sd and tuple(target_sd[nk].shape) == tuple(v.shape):
            out[nk] = v
    return out


def load_dlm_weights_into_msbd3(model, dlm_ckpt_path: str, verbose: bool = True) -> Tuple[list, list]:
    """Load (warm-start) a base DLM checkpoint into a ``MarlinDecoder`` model in-place."""
    ckpt = torch.load(dlm_ckpt_path, map_location="cpu", weights_only=False)
    dlm_sd = ckpt.get("state_dict", ckpt)
    target_sd = model.state_dict()
    remapped = remap_dlm_state_dict(dlm_sd, target_sd)
    missing, unexpected = model.load_state_dict(remapped, strict=False)

    if verbose:
        def nparams(keys):
            return sum(int(target_sd[k].numel()) for k in keys if k in target_sd)

        loaded_keys = [k for k in remapped if k in target_sd]
        total = sum(int(v.numel()) for v in target_sd.values())
        loaded = nparams(loaded_keys)
        # shared cross-attention aliases and computed buffers are not new
        truly_new = [k for k in missing
                     if not k.startswith("conditioner.cross_attention_layers.") and k != "m_heavy"]
        print(f"[warm-start] transferred {len(loaded_keys)} tensors "
              f"(~{loaded / 1e6:.1f}M / {total / 1e6:.1f}M params, {100 * loaded / total:.1f}%)")
        print(f"[warm-start] random-init (new heads): {sorted(set(k.split('.')[0] + '.' + k.split('.')[1] for k in truly_new))[:6]}")
        if unexpected:
            print(f"[warm-start] WARNING {len(unexpected)} unexpected keys (shape/name mismatch) — "
                  f"check that model dims match the checkpoint (hidden_size=896 for the released DLM).")
    return missing, unexpected


def _dry_run(args):
    from omegaconf import OmegaConf
    from marlin.model import MarlinDecoder

    cfg = OmegaConf.load(args.config)
    if args.hf_cache_dir:
        cfg.data.hf_cache_dir = args.hf_cache_dir
    model = MarlinDecoder(cfg)
    load_dlm_weights_into_msbd3(model, args.dlm_ckpt, verbose=True)
    print("[warm-start] dry-run complete (model not saved).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlm-ckpt", required=True)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "configs", "train_marlin_decoder.yaml"))
    ap.add_argument("--hf-cache-dir", default=None)
    _dry_run(ap.parse_args())
