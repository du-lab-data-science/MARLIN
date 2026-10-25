"""Spectrum-to-fingerprint encoders (``encoder.type``: ``mist`` or ``dreams``).

Both provide ``predict_proba(spec=..., batch=...)`` and a binarization ``threshold``.
"""

import os
import numpy as np
import torch


class MistEncoder:
    """Wraps MIST's SpectraEncoderGrowing; consumes the MIST featurized ``batch``."""

    name = "mist"
    mode = "fingerprint"  # produces a Morgan-fp probability vector

    def __init__(self, cfg, device, threshold=0.187):
        from mist.models import SpectraEncoderGrowing
        self.device = device
        self.threshold = float(threshold)
        c = cfg
        enc = SpectraEncoderGrowing(
            form_embedder=c.get("form_embedder", "pos-cos"),
            output_size=c.get("output_size", 4096),
            hidden_size=c.get("hidden_size", 512),
            spectra_dropout=c.get("spectra_dropout", 0.1),
            peak_attn_layers=c.get("peak_attn_layers", 2),
            num_heads=c.get("num_heads", 8),
            set_pooling=c.get("set_pooling", "cls"),
            refine_layers=c.get("refine_layers", 4),
            pairwise_featurization=c.get("pairwise_featurization", True),
            embed_instrument=c.get("embed_instrument", False),
            inten_transform=c.get("inten_transform", "float"),
            magma_modulo=c.get("magma_modulo", 2048),
            inten_prob=c.get("inten_prob", 0.1),
            remove_prob=c.get("remove_prob", 0.5),
            cls_type=c.get("cls_type", "ms1"),
            spec_features=c.get("spec_features", "peakformula"),
            mol_features=c.get("mol_features", "fingerprint"),
            top_layers=c.get("top_layers", 1),
        )
        ckpt_path = c["checkpoint"]
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            sd = ckpt.get("state_dict", ckpt)
            sd2 = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
            enc.load_state_dict(sd2 or sd, strict=False)
            print("[encoder] Loaded MIST encoder")
        else:
            print(f"[encoder] WARNING: MIST checkpoint not found: {ckpt_path}")
        self.model = enc.to(device).eval()

    @torch.no_grad()
    def predict_proba(self, spec=None, batch=None):
        probs, _ = self.model(batch)
        return probs.cpu().numpy()[0]


class DreamsEncoder:
    """Precomputed DreaMS embeddings, looked up by spectrum id, plus the fingerprint head."""

    name = "dreams"
    mode = "fingerprint"

    def __init__(self, cfg, device, default_threshold=0.5):
        from marlin.dreams_fphead import DreamsFPHead
        self.device = device

        emb_path = cfg["embeddings"]
        head_path = cfg["head_checkpoint"]
        data = np.load(emb_path, allow_pickle=True)
        ids = [str(x) for x in data["ids"]]
        embs = data["embs"].astype(np.float32)
        self.emb_kind = str(data["emb_kind"]) if "emb_kind" in data else "backbone"
        self.set_input = embs.ndim == 3   # all_peaks -> [N, P, D] (+ mask)
        mask = data["mask"].astype(np.float32) if "mask" in data else None
        self.cache = {sid: embs[i] for i, sid in enumerate(ids)}
        self.mcache = {sid: mask[i] for i, sid in enumerate(ids)} if mask is not None else None
        self.in_dim = embs.shape[-1]
        print(f"[encoder] Loaded DreaMS embeddings: {embs.shape} ({len(self.cache)} ids, emb_kind={self.emb_kind})")

        head, meta = DreamsFPHead.load(head_path, map_location=device)
        if meta.get("emb_kind", self.emb_kind) != self.emb_kind:
            print(f"[encoder] WARNING: head {head_path} emb_kind={meta.get('emb_kind')} != cache {self.emb_kind}")
        if head.in_dim != self.in_dim:
            raise ValueError(f"head {head_path} in_dim {head.in_dim} != embedding dim {self.in_dim}")
        self.head = head.to(device).eval()
        # the config's fp_threshold if set, else the threshold stored with the head
        self.threshold = float(cfg.get("fp_threshold") or float(meta.get("threshold", default_threshold)))
        self.bits = self.head.bits
        self._missing = 0
        print(f"[encoder] DreaMS FP head {head_path} loaded (val_tani={meta.get('val_tanimoto')}, "
              f"thr={self.threshold:.3f})")

    @torch.no_grad()
    def predict_proba(self, spec=None, batch=None):
        spec_id = spec.get_spec_name() if spec is not None else None
        emb = self.cache.get(spec_id)
        if emb is None:
            self._missing += 1
            if self._missing <= 5:
                print(f"[encoder] WARNING: no DreaMS embedding for '{spec_id}' "
                      f"(re-run dreams_precompute over this spectrum) -> zeros")
            return np.zeros(self.bits, dtype=np.float32)
        x = torch.from_numpy(emb).to(self.device).unsqueeze(0)
        m = None
        if self.mcache is not None:
            m = torch.from_numpy(self.mcache[spec_id]).to(self.device).unsqueeze(0)
        return self.head.predict_proba(x, m).cpu().numpy()[0]


def build_encoder(cfg, device):
    """MistEncoder or DreamsEncoder according to ``cfg['encoder']['type']`` (default ``mist``)."""
    enc_cfg = cfg.get("encoder", {}) or {}
    etype = enc_cfg.get("type", "mist")
    fp_thr = cfg.get("fingerprint", {}).get("threshold", 0.187)
    if etype == "mist":
        return MistEncoder(cfg["mist_encoder"], device, threshold=fp_thr)
    if etype == "dreams":
        return DreamsEncoder(enc_cfg, device, default_threshold=fp_thr)
    raise ValueError(f"Unknown encoder.type '{etype}' (expected 'mist' | 'dreams')")
