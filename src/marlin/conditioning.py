"""Mass, isotope and fingerprint conditioning, injected through shared cross-attention layers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from dlm.model_components import CrossAttentionLayer, FingerprintSequenceEncoder
from marlin.chem.mass_features import (
    ISOTOPE_FEATURE_DIM,
    fourier_mass_embed,
    fourier_mass_embed_dim,
)


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float, num_layers: int) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    for _ in range(max(num_layers - 2, 0)):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def _zero_init_last(mlp: nn.Sequential) -> None:
    """Zero the last layer so new conditioning starts as a no-op."""
    last = mlp[-1]
    if isinstance(last, nn.Linear):
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)


class MassConditioner(nn.Module):
    """Neutral mass -> one conditioning token."""

    def __init__(
        self,
        hidden_size: int,
        num_bands: int = 32,
        m_min: float = 0.0,
        m_max: float = 2000.0,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.m_min = m_min
        self.m_max = m_max
        self.mlp = _mlp(fourier_mass_embed_dim(num_bands), hidden_dim, hidden_size, dropout, num_layers)
        _zero_init_last(self.mlp)

    def forward(self, mass: torch.Tensor) -> torch.Tensor:
        emb = fourier_mass_embed(mass, self.num_bands, self.m_min, self.m_max)
        return self.mlp(emb).unsqueeze(1)


class IsotopeConditioner(nn.Module):
    """Isotope ratios -> one conditioning token."""

    def __init__(
        self,
        hidden_size: int,
        feature_dim: int = ISOTOPE_FEATURE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = _mlp(feature_dim, hidden_dim, hidden_size, dropout, num_layers)
        _zero_init_last(self.mlp)

    def forward(self, isotope: torch.Tensor) -> torch.Tensor:
        return self.mlp(isotope).unsqueeze(1)


class MassIsoFpConditioner(nn.Module):
    """Builds the conditioning sequence ``Z`` and owns the shared cross-attention layers."""

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
        *,
        use_mass: bool = True,
        use_isotope: bool = True,
        use_fingerprint: bool = True,
        mass_num_bands: int = 32,
        mass_min: float = 0.0,
        mass_max: float = 2000.0,
        mass_hidden_dim: int = 256,
        isotope_feature_dim: int = ISOTOPE_FEATURE_DIM,
        isotope_hidden_dim: int = 64,
        fingerprint_bits: int = 4096,
        fingerprint_max_seq_len: int = 256,
        fingerprint_num_self_attention_layers: int = 3,
        fingerprint_activation_threshold: float = 0.0,
        mass_dropout_prob: float = 0.1,
        isotope_dropout_prob: float = 0.1,
        fingerprint_dropout_prob: float = 0.25,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.fingerprint_bits = fingerprint_bits
        self.use_mass = use_mass
        self.use_isotope = use_isotope
        self.use_fingerprint = use_fingerprint
        self.mass_dropout_prob = mass_dropout_prob
        self.isotope_dropout_prob = isotope_dropout_prob
        self.fingerprint_dropout_prob = fingerprint_dropout_prob

        if use_mass:
            self.mass_conditioner = MassConditioner(
                hidden_size, num_bands=mass_num_bands, m_min=mass_min, m_max=mass_max,
                hidden_dim=mass_hidden_dim, dropout=dropout,
            )
        if use_isotope:
            self.isotope_conditioner = IsotopeConditioner(
                hidden_size, feature_dim=isotope_feature_dim, hidden_dim=isotope_hidden_dim, dropout=dropout,
            )
        if use_fingerprint:
            self.fingerprint_encoder = FingerprintSequenceEncoder(
                num_bits=fingerprint_bits,
                embedding_dim=hidden_size,
                max_seq_len=fingerprint_max_seq_len,
                activation_threshold=fingerprint_activation_threshold,
                dropout=dropout,
                num_self_attention_layers=fingerprint_num_self_attention_layers,
                num_attention_heads=num_attention_heads,
            )

        # one cross-attention layer per transformer block
        self.cross_attention_layers = nn.ModuleList(
            [CrossAttentionLayer(hidden_size, num_attention_heads, dropout) for _ in range(num_layers)]
        )

    def _standardize_fp(self, fingerprint, batch_size: int, device) -> torch.Tensor:
        if fingerprint is None:
            return torch.zeros(batch_size, self.fingerprint_bits, device=device, dtype=torch.float32)
        fp = fingerprint if torch.is_tensor(fingerprint) else torch.as_tensor(fingerprint, dtype=torch.float32)
        fp = fp.to(device=device, dtype=torch.float32)
        if fp.dim() == 1:
            fp = fp.unsqueeze(0)
        if fp.size(0) == 1 and batch_size > 1:
            fp = fp.expand(batch_size, -1)
        return fp

    def _standardize_mass(self, mass, batch_size: int, device) -> torch.Tensor:
        if mass is None:
            return torch.zeros(batch_size, device=device, dtype=torch.float32)
        m = mass if torch.is_tensor(mass) else torch.as_tensor(mass, dtype=torch.float32)
        m = m.to(device=device, dtype=torch.float32).reshape(-1)
        if m.numel() == 1 and batch_size > 1:
            m = m.expand(batch_size)
        return m

    def _standardize_iso(self, isotope, batch_size: int, device) -> torch.Tensor:
        dim = getattr(self.isotope_conditioner, "feature_dim", ISOTOPE_FEATURE_DIM) if self.use_isotope else ISOTOPE_FEATURE_DIM
        if isotope is None:
            return torch.zeros(batch_size, dim, device=device, dtype=torch.float32)
        it = isotope if torch.is_tensor(isotope) else torch.as_tensor(isotope, dtype=torch.float32)
        it = it.to(device=device, dtype=torch.float32)
        if it.dim() == 1:
            it = it.unsqueeze(0)
        if it.size(0) == 1 and batch_size > 1:
            it = it.expand(batch_size, -1)
        return it

    def encode(
        self,
        batch_size: int,
        device,
        mass: Optional[torch.Tensor] = None,
        isotope: Optional[torch.Tensor] = None,
        fingerprint: Optional[torch.Tensor] = None,
        fingerprint_mask: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return ``(Z [B, S, H], Z_mask [B, S])``.

        Disabled inputs are omitted; missing or randomly dropped inputs become zero tokens with mask 0.
        """
        ones = lambda: torch.ones(batch_size, 1, device=device, dtype=torch.float32)
        zeros = lambda: torch.zeros(batch_size, 1, device=device, dtype=torch.float32)
        embs = []
        masks = []

        if self.use_mass:
            m = self._standardize_mass(mass, batch_size, device)
            z_m = self.mass_conditioner(m)
            keep = mass is not None
            if training and keep and torch.rand(1).item() < self.mass_dropout_prob:
                keep = False
            if not keep:
                z_m = z_m * 0.0
            embs.append(z_m)
            masks.append(ones() if keep else zeros())

        if self.use_isotope:
            it = self._standardize_iso(isotope, batch_size, device)
            z_iso = self.isotope_conditioner(it)
            keep = isotope is not None
            if training and keep and torch.rand(1).item() < self.isotope_dropout_prob:
                keep = False
            if not keep:
                z_iso = z_iso * 0.0
            embs.append(z_iso)
            masks.append(ones() if keep else zeros())

        if self.use_fingerprint:
            fp = self._standardize_fp(fingerprint, batch_size, device)
            z_f, m_f = self.fingerprint_encoder(fp)
            keep = fingerprint is not None
            if training and keep and torch.rand(1).item() < self.fingerprint_dropout_prob:
                keep = False
            if not keep:
                z_f = z_f * 0.0
                m_f = torch.zeros_like(m_f)
            # skip samples whose fingerprint is flagged invalid
            if fingerprint_mask is not None:
                fm = self._standardize_mass(fingerprint_mask, batch_size, device).clamp(0, 1).view(-1, 1)
                z_f = z_f * fm.unsqueeze(-1)
                m_f = m_f * fm
            embs.append(z_f)
            masks.append(m_f)

        if not embs:
            return None, None
        Z = torch.cat(embs, dim=1)
        Z_mask = torch.cat(masks, dim=1)
        return Z, Z_mask
