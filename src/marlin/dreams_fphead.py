"""Fingerprint head on DreaMS embeddings: an MLP (``iterative='none'``) or MIST's growing head (``'growing'``)."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class _FPGrowing(nn.Module):
    """MIST's FPGrowingModule: predicts the fingerprint coarse to fine and returns every level."""

    def __init__(self, hidden_input_dim, final_target_dim=4096, num_splits=4, reduce_factor=2):
        super().__init__()
        dims = [int(np.ceil(final_target_dim / (reduce_factor ** k))) for k in range(num_splits + 1)][::-1]
        self.output_dims = dims
        self.initial_predict = nn.Sequential(nn.Linear(hidden_input_dim, dims[0]), nn.Sigmoid())
        pb, gb = [], []
        for i, d in enumerate(dims[:-1]):
            od = dims[i + 1]
            pb.append(nn.Sequential(nn.Linear(d, od), nn.Sigmoid()))        # grow prev prediction
            gb.append(nn.Sequential(nn.Linear(hidden_input_dim, od), nn.Sigmoid()))  # gate from hidden
        self.predict_bricks = nn.ModuleList(pb)
        self.gate_bricks = nn.ModuleList(gb)

    def forward(self, hidden):
        cur = self.initial_predict(hidden)
        outs = [cur]
        for pb, gb in zip(self.predict_bricks, self.gate_bricks):
            cur = gb(hidden) * pb(cur)
            outs.append(cur)
        return outs  # outs[-1] is the full fingerprint


class DreamsFPHead(nn.Module):
    """DreaMS embedding, or a set of peak embeddings, -> per-bit fingerprint probabilities."""

    def __init__(self, in_dim: int, bits: int = 4096, hidden_dim: int = 2048,
                 depth: int = 2, dropout: float = 0.1, set_input: bool = False,
                 iterative: str = "none", num_splits: int = 4, reduce_factor: int = 2):
        super().__init__()
        self.in_dim = int(in_dim)
        self.bits = int(bits)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.dropout = float(dropout)
        self.set_input = bool(set_input)
        self.iterative = str(iterative)
        self.num_splits = int(num_splits)
        self.reduce_factor = int(reduce_factor)

        if self.set_input:
            self.pool_q = nn.Parameter(torch.zeros(self.in_dim))   # learned attention query
            nn.init.normal_(self.pool_q, std=0.02)

        layers, d = [], self.in_dim
        for _ in range(max(self.depth - 1, 0)):
            layers += [nn.Linear(d, self.hidden_dim), nn.LayerNorm(self.hidden_dim),
                       nn.GELU(), nn.Dropout(self.dropout)]
            d = self.hidden_dim
        if self.iterative == "growing":
            self.trunk = nn.Sequential(*layers) if layers else nn.Identity()
            self.grow = _FPGrowing(d, self.bits, self.num_splits, self.reduce_factor)
        else:
            layers.append(nn.Linear(d, self.bits))  # logits
            self.net = nn.Sequential(*layers)

    def _pool(self, x, mask):
        # masked attention pooling over peaks
        scores = (x * self.pool_q).sum(-1) / (self.in_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask < 0.5, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (w * x).sum(1)

    def forward(self, x, mask=None):
        if self.set_input or x.dim() == 3:
            x = self._pool(x, mask)
        if self.iterative == "growing":
            return self.grow(self.trunk(x))    # list of sigmoid probs (final + intermediates)
        return self.net(x)                     # logits

    def forward_all(self, x, mask=None):
        """All sigmoid predictions of the growing head (for the iterative loss)."""
        out = self.forward(x, mask)
        return out if isinstance(out, list) else [torch.sigmoid(out)]

    @torch.no_grad()
    def predict_proba(self, x, mask=None):
        out = self.forward(x, mask)
        return out[-1] if isinstance(out, list) else torch.sigmoid(out)  # growing list is already sigmoid

    # checkpoint: weights, architecture and the tuned threshold
    def save(self, path: str, emb_kind: str = "backbone", threshold: float = 0.5, extra: Optional[dict] = None):
        torch.save({
            "state_dict": self.state_dict(),
            "meta": {
                "in_dim": self.in_dim, "bits": self.bits, "hidden_dim": self.hidden_dim,
                "depth": self.depth, "dropout": self.dropout, "set_input": self.set_input,
                "iterative": self.iterative, "num_splits": self.num_splits, "reduce_factor": self.reduce_factor,
                "emb_kind": emb_kind, "threshold": float(threshold),
                **(extra or {}),
            },
        }, path)

    @classmethod
    def load(cls, path: str, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        m = ckpt["meta"]
        head = cls(in_dim=m["in_dim"], bits=m["bits"], hidden_dim=m["hidden_dim"],
                   depth=m["depth"], dropout=m["dropout"], set_input=m.get("set_input", False),
                   iterative=m.get("iterative", "none"), num_splits=m.get("num_splits", 4),
                   reduce_factor=m.get("reduce_factor", 2))
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        return head, m
