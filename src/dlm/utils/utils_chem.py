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


import safe as sf
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# https://github.com/datamol-io/safe/blob/main/safe/sample.py
def safe_to_smiles(safe_str, fix=True):
    if fix:
        safe_str = '.'.join([frag for frag in safe_str.split('.')
                             if sf.decode(frag, ignore_errors=True) is not None])
    return sf.decode(safe_str, canonical=True, ignore_errors=True)

_SAFE_CONVERTER = sf.SAFEConverter(slicer=None, ignore_stereo=True)
def smiles_to_safe(smiles: str) -> str:
    """Convert SMILES to SAFE representation."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        safe_str = _SAFE_CONVERTER.encoder(smiles, allow_empty=True)
        return safe_str
    except Exception as e:
        print(f"Failed to convert SMILES to SAFE: {smiles}, error: {e}")
        return None
