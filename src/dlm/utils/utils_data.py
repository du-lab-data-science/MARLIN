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


import os

os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import datasets
from safe.tokenizer import SAFETokenizer
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def get_last_checkpoint(save_dir):
    if not save_dir or not os.path.exists(save_dir):
        return None

    candidates = []
    for filename in os.listdir(save_dir):
        if not filename.endswith('.ckpt'):
            continue
        stem = filename[:-5]
        try:
            step = int(stem)
        except ValueError:
            continue
        candidates.append((step, filename))

    if not candidates:
        return None

    _, last_filename = max(candidates, key=lambda x: x[0])
    return os.path.join(save_dir, last_filename)
    

def get_tokenizer(hf_cache_dir=None):
    try:
        tk = SAFETokenizer.from_pretrained(
            'datamol-io/safe-gpt',
            cache_dir=hf_cache_dir,
            local_files_only=True  # Force using only local cached files, if throws error, set this to False and then re-run
        ).get_pretrained()
    except Exception as e:
        print("Error loading tokenizer from HF cache:", e)
        tk = SAFETokenizer.from_pretrained(
            'datamol-io/safe-gpt',
            cache_dir=hf_cache_dir,
            local_files_only=False
        ).get_pretrained()
    tk.add_tokens(['<', '>'])   # for bracket_safe
    return tk
    

class UserDataset(datasets.Dataset):
    def __init__(self, data_path):
        if not data_path:
            raise ValueError(
                "config.data.data_path must be set when data.use_safe_hf is False."
            )
        data_path = os.fspath(data_path)
        with open(data_path) as f:
            self.safe_list = f.readlines()
        self.safe_list = [s.strip('\n') for s in self.safe_list]
        
    def __len__(self):
        return len(self.safe_list)

    def __getitem__(self, indices):
        return {'input': self.safe_list[i] for i in indices}
