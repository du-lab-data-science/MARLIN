"""Per-layer key/value cache over finished blocks for faster sampling.

Under block-causal attention a finished block's keys and values never change, so the cached forward
gives the same logits as recomputing the prefix with ``model.denoise_single``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

NEG = -1e9


class BlockKVCache:
    """Incremental per-layer K/V cache over committed (clean) blocks."""

    def __init__(self, model, num_candidates: int, device, cond: Tuple):
        self.model = model
        self.N = int(num_candidates)
        self.device = device
        self.cond = cond  # (Z, Z_mask) or (None, None)

        bert = model.backbone.bert
        self.embeddings = bert.embeddings
        self.layers = bert.encoder.layer
        self.cls = model.backbone.cls
        self.n_layers = len(self.layers)

        self.cache_k: List[Optional[torch.Tensor]] = [None] * self.n_layers
        self.cache_v: List[Optional[torch.Tensor]] = [None] * self.n_layers
        self.prefix_valid: Optional[torch.Tensor] = None  # True for non-pad keys

    # matches model._token_embeddings plus the embedding LayerNorm
    def _embed(self, ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        e = self.embeddings
        h = e.word_embeddings(ids) + e.position_embeddings(positions) \
            + e.token_type_embeddings(torch.zeros_like(ids))
        h = e.LayerNorm(h)
        h = e.dropout(h)  # eval -> identity
        return h

    def _block_forward(self, ids: torch.Tensor, positions: torch.Tensor, block_valid: torch.Tensor):
        """Forward one block against the cache; returns ``(logits, k, v)`` with this block's keys and values."""
        Z, Z_mask = self.cond
        N, Bp = ids.shape
        h = self._embed(ids, positions)
        new_k: List[torch.Tensor] = []
        new_v: List[torch.Tensor] = []

        for i, layer in enumerate(self.layers):
            sa = layer.bert_layer.attention.self
            q = sa.transpose_for_scores(sa.query(h))
            kb = sa.transpose_for_scores(sa.key(h))
            vb = sa.transpose_for_scores(sa.value(h))

            if self.cache_k[i] is not None:
                k = torch.cat([self.cache_k[i], kb], dim=2)
                v = torch.cat([self.cache_v[i], vb], dim=2)
                valid = torch.cat([self.prefix_valid, block_valid], dim=1)
            else:
                k, v = kb, vb
                valid = block_valid

            scale = math.sqrt(sa.attention_head_size)
            scores = torch.matmul(q, k.transpose(-1, -2)) / scale
            add = torch.where(valid.bool(), torch.zeros((), device=self.device),
                              torch.full((), NEG, device=self.device)).view(N, 1, 1, -1)
            scores = scores + add
            probs = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(probs, v)
            ctx = ctx.permute(0, 2, 1, 3).contiguous().view(N, Bp, sa.all_head_size)

            bl = layer.bert_layer
            attn_out = bl.attention.output(ctx, h)     # dense + dropout + LayerNorm(+h)
            inter = bl.intermediate(attn_out)
            h = bl.output(inter, attn_out)             # dense + dropout + LayerNorm(+attn_out)
            if layer.has_cross_attention and Z is not None:
                h = layer.cross_attention(h, Z, Z_mask)

            new_k.append(kb)
            new_v.append(vb)

        logits = self.cls(h)
        return logits, new_k, new_v

    @torch.no_grad()
    def decode(self, ids: torch.Tensor, positions: torch.Tensor, block_valid: torch.Tensor) -> torch.Tensor:
        """Block logits ``[N, B', V]`` (does NOT modify the cache)."""
        with self.model._autocast():
            logits, _, _ = self._block_forward(ids, positions, block_valid)
        return logits

    @torch.no_grad()
    def commit(self, ids: torch.Tensor, positions: torch.Tensor, block_valid: torch.Tensor) -> None:
        """Append a finalized (clean) block's K/V to the cache."""
        with self.model._autocast():
            _, new_k, new_v = self._block_forward(ids, positions, block_valid)
        for i in range(self.n_layers):
            if self.cache_k[i] is None:
                self.cache_k[i] = new_k[i]
                self.cache_v[i] = new_v[i]
            else:
                self.cache_k[i] = torch.cat([self.cache_k[i], new_k[i]], dim=2)
                self.cache_v[i] = torch.cat([self.cache_v[i], new_v[i]], dim=2)
        self.prefix_valid = block_valid if self.prefix_valid is None \
            else torch.cat([self.prefix_valid, block_valid], dim=1)
