"""DreaMS embedding adapter (runs in the DreaMS environment).

Checkpoints default to ``weights/dreams/{ssl_model,embedding_model}.ckpt`` (official DreaMS weights,
https://zenodo.org/records/10997887) and can be overridden with ``$DREAMS_SSL_CKPT`` / ``$DREAMS_EMB_CKPT``.
Set ``$DREAMS_DIR`` to a DreaMS source checkout if the ``dreams`` package is not installed.
"""

import os
import sys
import types
import importlib
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

_RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DREAMS_DIR = os.environ.get("DREAMS_DIR")  # optional DreaMS source checkout (if not pip-installed)

_DREAMS_WEIGHTS = os.path.join(_RELEASE_ROOT, "weights", "dreams")
DEFAULT_SSL_CKPT = os.environ.get("DREAMS_SSL_CKPT", os.path.join(_DREAMS_WEIGHTS, "ssl_model.ckpt"))
DEFAULT_EMB_CKPT = os.environ.get("DREAMS_EMB_CKPT", os.path.join(_DREAMS_WEIGHTS, "embedding_model.ckpt"))

_PREPROC = None  # cached SpectrumPreprocessor


def _prepare_dreams_imports():
    """Make ``import dreams.*`` work for the embedding path without its unused heavy extras."""
    # DreaMS imports these at load time; the embedding path never uses them
    for stub in ("umap", "pynndescent"):
        if stub not in sys.modules:
            try:
                __import__(stub)
            except Exception:
                sys.modules[stub] = types.ModuleType(stub)
    if DREAMS_DIR and DREAMS_DIR not in sys.path:
        sys.path.insert(0, DREAMS_DIR)


def _alias_legacy_msml(module_name):
    """Alias the legacy ``msml.*`` module paths of old checkpoints to ``dreams.*``."""
    if not module_name or not (module_name == "msml" or module_name.startswith("msml.")):
        return False
    mapped = "dreams" + module_name[len("msml"):]
    try:
        m = importlib.import_module(mapped)
    except ModuleNotFoundError:
        return False
    parent = module_name.rpartition(".")[0]
    if parent and parent not in sys.modules:
        _alias_legacy_msml(parent)
    legacy = types.ModuleType(module_name)
    legacy.__dict__.update(m.__dict__)
    legacy.__name__ = module_name
    if hasattr(m, "__path__"):
        legacy.__path__ = m.__path__
    sys.modules[module_name] = legacy
    if parent in sys.modules:
        setattr(sys.modules[parent], module_name.split(".")[-1], legacy)
    return True


def _load_from_checkpoint(cls, path, **kwargs):
    """``cls.load_from_checkpoint`` with legacy-module-alias retries."""
    for _ in range(12):
        try:
            return cls.load_from_checkpoint(path, map_location="cpu", **kwargs)
        except ModuleNotFoundError as e:
            if not _alias_legacy_msml(getattr(e, "name", None)):
                raise
    raise RuntimeError(f"Failed to load DreaMS checkpoint after alias retries: {path}")


class DreamsEmbedder:
    """Frozen DreaMS encoder: (peaks, precursor m/z) -> embedding [d_model]."""

    def __init__(self, emb_ckpt=DEFAULT_EMB_CKPT, ssl_ckpt=DEFAULT_SSL_CKPT,
                 emb_kind="backbone", n_highest_peaks=100, device=None,
                 finetuned_backbone=None):
        """``emb_kind``: ``backbone`` (precursor token), ``contrastive`` or ``all_peaks``;
        ``finetuned_backbone``: backbone weights saved by ``finetune_dreams_fp_backbone.py``."""
        _prepare_dreams_imports()
        from dreams.models.heads.heads import ContrastiveHead  # noqa: E402

        assert emb_kind in ("backbone", "contrastive", "all_peaks")
        self.emb_kind = emb_kind
        self.n_highest_peaks = n_highest_peaks
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        head = _load_from_checkpoint(ContrastiveHead, emb_ckpt, backbone_pth=Path(ssl_ckpt))
        self.head = head.eval().to(self.device)
        self.backbone = self.head.backbone
        self.d_model = int(self.backbone.d_model)

        if finetuned_backbone:
            ckpt = torch.load(finetuned_backbone, map_location=self.device)
            sd = ckpt.get("state_dict", ckpt)
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            self.backbone = self.backbone.eval().to(self.device)
            print(f"[dreams] loaded fine-tuned backbone {finetuned_backbone} "
                  f"(missing={len(missing)} unexpected={len(unexpected)})")

    def _preprocessor(self):
        global _PREPROC
        if _PREPROC is None:
            from dreams.utils.data import SpectrumPreprocessor
            from dreams.utils.dformats import DataFormatA
            _PREPROC = SpectrumPreprocessor(dformat=DataFormatA(), n_highest_peaks=self.n_highest_peaks)
        return _PREPROC

    def preprocess(self, peaks, prec_mz):
        """peaks: array (n,2) or (2,n) of [m/z, intensity] -> SPECTRUM tensor (n_peaks+1, 2)."""
        pp = self._preprocessor()
        spec = np.asarray(peaks, dtype=np.float64)
        # tell the preprocessor the orientation of the peak array
        high_form = spec.shape[-1] == 2 and spec.shape[0] != 2
        out = pp(spec, prec_mz=float(prec_mz), high_form=high_form if spec.shape != (2, 2) else True)
        return np.asarray(out, dtype=np.float32)

    @torch.no_grad()
    def embed(self, specs, batch_size=64):
        """Embed ``[(peaks, prec_mz), ...]``; returns ``(embs, mask)``, with a mask only for ``all_peaks``."""
        tensors = [torch.from_numpy(self.preprocess(p, m)) for p, m in specs]
        N = len(tensors)
        if self.emb_kind == "all_peaks":
            P = tensors[0].shape[0]
            embs = np.zeros((N, P, self.d_model), dtype=np.float16)
            mask = np.zeros((N, P), dtype=np.float32)
            for i in range(0, N, batch_size):
                x = torch.stack(tensors[i:i + batch_size]).to(self.device, dtype=self.backbone.dtype)
                full = self.backbone(x, None)
                embs[i:i + x.shape[0]] = full.float().cpu().numpy().astype(np.float16)
                mask[i:i + x.shape[0]] = (x[:, :, 0] != 0).float().cpu().numpy()  # m/z==0 -> padding
            return embs, mask
        out = np.zeros((N, self.d_model), dtype=np.float32)
        for i in range(0, N, batch_size):
            x = torch.stack(tensors[i:i + batch_size]).to(self.device, dtype=self.backbone.dtype)
            emb = self.head(x, no_head=True) if self.emb_kind == "backbone" else self.head(x)
            out[i:i + x.shape[0]] = emb.float().cpu().numpy()
        return out, None
