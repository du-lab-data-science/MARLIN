"""Neutral mass from the precursor, isotope ratios, and the Fourier mass embedding."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch

from mist.utils.chem_utils import (  # noqa: F401
    ELECTRON_MASS,
    ELEMENT_TO_MASS,
    ion_to_mass,
    standardize_adduct,
)

PROTON_MASS = 1.0072764665789

# Natural abundance of the +1 / +2 isotope per atom, relative to the most abundant isotope.
_M1_PER_ATOM: Dict[str, float] = {
    "C": 0.010816,   # 13C / 12C
    "N": 0.003693,   # 15N / 14N
    "O": 0.000381,   # 17O / 16O
    "S": 0.007893,   # 33S / 32S
    "Si": 0.050800,  # 29Si / 28Si
    "H": 0.000115,   # 2H / 1H
}
_M2_PER_ATOM: Dict[str, float] = {
    "O": 0.002005,   # 18O / 16O
    "S": 0.044306,   # 34S / 32S
    "Si": 0.033872,  # 30Si / 28Si
    "Cl": 0.319961,  # 37Cl / 35Cl
    "Br": 0.972775,  # 81Br / 79Br
}

ISOTOPE_FEATURE_DIM = 2


def isotope_features_from_counts(counts: Dict[str, int]) -> torch.Tensor:
    """Approximate ``[I(M+1)/I(M), I(M+2)/I(M)]`` from element counts."""
    r1 = sum(_M1_PER_ATOM.get(el, 0.0) * float(n) for el, n in counts.items())
    r2 = sum(_M2_PER_ATOM.get(el, 0.0) * float(n) for el, n in counts.items())
    # M+2 from two 13C atoms
    n_c = float(counts.get("C", 0))
    r2 += 0.5 * (_M1_PER_ATOM["C"] * n_c) ** 2
    return torch.tensor([r1, r2], dtype=torch.float32)


def isotope_features_from_smiles(smiles: Optional[str]) -> torch.Tensor:
    """Isotope features computed from a SMILES (decoder training)."""
    if not smiles:
        return torch.zeros(ISOTOPE_FEATURE_DIM, dtype=torch.float32)
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(ISOTOPE_FEATURE_DIM, dtype=torch.float32)
    mol_h = Chem.AddHs(mol)
    counts: Dict[str, int] = {}
    for atom in mol_h.GetAtoms():
        sym = atom.GetSymbol()
        counts[sym] = counts.get(sym, 0) + 1
    return isotope_features_from_counts(counts)


# m/z spacing between adjacent isotope peaks
_ISOTOPE_SPACING = 1.003355


def isotope_features_from_spectrum(peaks, precursor_mz: float, tol_da: float = 0.02):
    """Isotope ratios from measured ``(m/z, intensity)`` peaks; ``None`` without a precursor peak."""
    if peaks is None or precursor_mz is None or precursor_mz <= 0:
        return None
    arr = np.asarray(peaks, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
        return None
    mzs, intens = arr[:, 0], arr[:, 1]

    def peak_intensity(center):
        sel = np.abs(mzs - center) <= tol_da
        return float(intens[sel].max()) if np.any(sel) else 0.0

    i0 = peak_intensity(precursor_mz)
    if i0 <= 0:
        return None
    i1 = peak_intensity(precursor_mz + _ISOTOPE_SPACING)
    i2 = peak_intensity(precursor_mz + 2 * _ISOTOPE_SPACING)
    return torch.tensor([i1 / i0, i2 / i0], dtype=torch.float32)


def neutral_mass_from_precursor(mz: float, adduct: str, charge: int = 1) -> Optional[float]:
    """Neutral mass ``M = m/z * |z| - adduct mass``; ``None`` for an unknown adduct."""
    try:
        adduct_std = standardize_adduct(adduct)
    except Exception:
        return None
    delta = ion_to_mass.get(adduct_std)
    if delta is None or mz is None or mz <= 0:
        return None
    z = abs(int(charge)) if charge else 1
    return float(mz) * z - float(delta)


def fourier_mass_embed(
    mass: torch.Tensor,
    num_bands: int = 32,
    m_min: float = 0.0,
    m_max: float = 2000.0,
) -> torch.Tensor:
    """Sinusoidal embedding of a scalar mass, ``[B] -> [B, 2 * num_bands + 1]``."""
    if mass.dim() == 2 and mass.size(-1) == 1:
        mass = mass.squeeze(-1)
    x = (mass - m_min) / max(m_max - m_min, 1e-6)
    x = x.clamp(0.0, 1.0).unsqueeze(-1)
    device, dtype = mass.device, torch.float32
    # frequencies pi * 2^k, k in [0, 10]
    exponents = torch.linspace(0.0, math.log2(1024.0), num_bands, device=device, dtype=dtype)
    freqs = (2.0 ** exponents) * math.pi
    ang = x.to(dtype) * freqs
    emb = torch.cat([torch.sin(ang), torch.cos(ang), x.to(dtype)], dim=-1)
    return emb


def fourier_mass_embed_dim(num_bands: int = 32) -> int:
    return 2 * num_bands + 1
