# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Formula normalization and Morgan-fingerprint helpers (the subset of FRIGID's
benchmark utilities that MARLIN uses).
"""
import re
from typing import List, Dict, Optional
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

# Element pattern for Hill notation
ELEMENT_PATTERN = re.compile(r'([A-Z][a-z]?)(\d*)')


# =============================================================================
# Formula Functions
# =============================================================================

def normalize_formula(formula: Optional[str]) -> Optional[str]:
    """
    Normalize molecular formula into Hill notation for robust string comparison.
    
    Hill notation: C first, H second (if C present), then alphabetical.
    
    Args:
        formula: Molecular formula string
        
    Returns:
        Normalized formula string or None if invalid
    """
    if not formula:
        return None

    matches = ELEMENT_PATTERN.findall(str(formula))
    if not matches:
        return formula

    counts: Dict[str, int] = {}
    for element, count_str in matches:
        if element:
            count = int(count_str) if count_str else 1
            counts[element] = counts.get(element, 0) + count

    # Hill notation ordering
    ordered_elements: List[str] = []
    if 'C' in counts:
        ordered_elements.append('C')
        if 'H' in counts:
            ordered_elements.append('H')

    ordered_elements.extend(
        sorted(elem for elem in counts.keys() if elem not in ordered_elements)
    )

    normalized = ''.join(
        f"{elem}{counts[elem] if counts[elem] != 1 else ''}"
        for elem in ordered_elements
    )
    return normalized


def compute_morgan_fingerprint(
    smiles: str, 
    n_bits: int = 4096, 
    radius: int = 2
) -> Optional[np.ndarray]:
    """
    Compute Morgan fingerprint as numpy array.
    
    Args:
        smiles: SMILES string
        n_bits: Number of fingerprint bits
        radius: Morgan fingerprint radius
        
    Returns:
        Fingerprint as numpy array or None if failed
    """
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


def compute_tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Compute Tanimoto similarity between two fingerprint arrays."""
    if fp1 is None or fp2 is None:
        return 0.0
    intersection = np.sum(np.minimum(fp1, fp2))
    union = np.sum(np.maximum(fp1, fp2))
    return float(intersection / union) if union > 0 else 0.0


def get_inchikey_first_block(inchi_key: Optional[str]) -> Optional[str]:
    """Extract first block of InChI key (connectivity layer, no stereo)."""
    if not inchi_key:
        return None
    return inchi_key.strip().upper().split('-')[0]


def binarize_fingerprint(fp_probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binarize fingerprint probabilities."""
    return (fp_probs >= threshold).astype(np.float32)
