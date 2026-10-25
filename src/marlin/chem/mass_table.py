"""Per-token heavy-atom masses and grammar/valence tables for mass-shell decoding.

Hydrogens are left out: their count is fixed only once the molecule is complete, so the
heavy-atom mass of a partial molecule is a lower bound on its final mass.
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch

from dlm.utils.formula_encoder import FormulaEncoder
from mist.utils.chem_utils import ELEMENT_TO_MASS

_RD_PTBL = None


def _periodic_table():
    global _RD_PTBL
    if _RD_PTBL is None:
        from rdkit import Chem

        _RD_PTBL = Chem.GetPeriodicTable()
    return _RD_PTBL


def element_mono_mass(symbol: str) -> float:
    """Monoisotopic mass of an element (MIST's table, RDKit as fallback)."""
    if symbol in ELEMENT_TO_MASS:
        return float(ELEMENT_TO_MASS[symbol])
    try:
        return float(_periodic_table().GetMostCommonIsotopeMass(symbol))
    except Exception:
        return 0.0


def _token_heavy_mass(token_str: str, encoder: FormulaEncoder) -> float:
    """Heavy-atom monoisotopic mass of one decoded token."""
    if not token_str:
        return 0.0
    try:
        counts = encoder.parse_atoms_from_smiles(token_str)
    except Exception:
        return 0.0
    total = 0.0
    for el, n in counts.items():
        if el == "H":
            continue
        total += element_mono_mass(el) * float(n)
    return total


def build_mass_table(tokenizer, special_token_ids: Optional[List[int]] = None) -> torch.Tensor:
    """Heavy-atom mass of every token, ``[vocab_size]``; special tokens get 0."""
    encoder = FormulaEncoder(normalize="none")
    vocab_size = tokenizer.vocab_size

    if special_token_ids is None:
        special_token_ids = []
        for attr in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "mask_token_id",
            "cls_token_id",
            "sep_token_id",
            "unk_token_id",
        ):
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                special_token_ids.append(int(tid))
    special = set(int(t) for t in special_token_ids)

    masses = torch.zeros(vocab_size, dtype=torch.float64)
    for tid in range(vocab_size):
        if tid in special:
            continue
        token_str = tokenizer.decode([tid])
        masses[tid] = _token_heavy_mass(token_str, encoder)
    return masses.to(torch.float32)


# Neutral valences for bounding the hydrogens of a partial molecule: n connected heavy
# atoms carry at most sum(valence) - 2 * (n - 1) hydrogens.
_ELEMENT_MAX_H_VALENCE = {
    "C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "B": 3, "Si": 4, "Ge": 4,
    "Se": 2, "Te": 2, "As": 3, "Sb": 3,
    "F": 1, "Cl": 1, "Br": 1, "I": 1, "At": 1,
}
_DEFAULT_VALENCE = 4  # rare and metal elements


def build_token_grammar(tokenizer, special_token_ids=None):
    """Per-token change of the branch balance and its minimum within the token, ``[V]`` each."""
    vocab_size = tokenizer.vocab_size
    if special_token_ids is None:
        special_token_ids = []
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "mask_token_id",
                     "cls_token_id", "sep_token_id", "unk_token_id"):
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                special_token_ids.append(int(tid))
    special = set(int(t) for t in special_token_ids)

    paren_delta = torch.zeros(vocab_size, dtype=torch.float32)
    paren_min = torch.zeros(vocab_size, dtype=torch.float32)
    for tid in range(vocab_size):
        if tid in special:
            continue
        s = tokenizer.decode([tid])
        bal = 0
        mn = 0
        for ch in s:
            if ch == "(":
                bal += 1
            elif ch == ")":
                bal -= 1
                mn = min(mn, bal)
        paren_delta[tid] = float(bal)
        paren_min[tid] = float(mn)
    return paren_delta, paren_min


def build_valence_table(tokenizer, special_token_ids=None):
    """Per-token sum of heavy-atom valences and heavy-atom count, ``[V]`` each."""
    encoder = FormulaEncoder(normalize="none")
    vocab_size = tokenizer.vocab_size
    if special_token_ids is None:
        special_token_ids = []
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "mask_token_id",
                     "cls_token_id", "sep_token_id", "unk_token_id"):
            tid = getattr(tokenizer, attr, None)
            if tid is not None:
                special_token_ids.append(int(tid))
    special = set(int(t) for t in special_token_ids)

    val_sum = torch.zeros(vocab_size, dtype=torch.float32)
    n_heavy = torch.zeros(vocab_size, dtype=torch.float32)
    for tid in range(vocab_size):
        if tid in special:
            continue
        try:
            counts = encoder.parse_atoms_from_smiles(tokenizer.decode([tid]))
        except Exception:
            continue
        vs = 0.0
        nh = 0.0
        for el, c in counts.items():
            if el == "H":
                continue
            vs += _ELEMENT_MAX_H_VALENCE.get(el, _DEFAULT_VALENCE) * float(c)
            nh += float(c)
        val_sum[tid] = vs
        n_heavy[tid] = nh
    return val_sum, n_heavy


def min_positive_token_mass(m_heavy: torch.Tensor) -> float:
    """Smallest positive token mass."""
    pos = m_heavy[m_heavy > 0]
    if pos.numel() == 0:
        return 0.0
    return float(pos.min().item())


def save_mass_table(m_heavy: torch.Tensor, path: str, tokenizer=None) -> None:
    """Save the mass table as parquet."""
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = {"token_id": list(range(m_heavy.numel())), "m_heavy": m_heavy.tolist()}
    if tokenizer is not None:
        rows["token_str"] = [tokenizer.decode([i]) for i in range(m_heavy.numel())]
    pd.DataFrame(rows).to_parquet(path, index=False)
