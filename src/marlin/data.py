"""Decoder training data: SAFE-GPT strings with mass, isotope and fingerprint conditioning from RDKit."""

from __future__ import annotations

import glob
import os

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

from dlm.utils.utils_chem import safe_to_smiles, smiles_to_safe
from dlm.utils.utils_data import UserDataset, get_tokenizer
from mist.utils.chem_utils import mass_from_smi
from marlin.chem.mass_features import (
    ISOTOPE_FEATURE_DIM,
    isotope_features_from_smiles,
)

RDLogger.DisableLog("rdApp.*")


def _rank_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


class MSBD3Collator:
    """Collate SAFE strings into ``input_ids`` + mass/isotope/fingerprint."""

    def __init__(self, config):
        self.tokenizer = get_tokenizer(config.data.get("hf_cache_dir", None))
        self.remove_stereo = config.data.get("remove_stereo", False)
        self.max_length = config.model.max_position_embeddings
        m = config.model
        self.use_mass = m.get("use_mass_conditioning", True)
        self.use_isotope = m.get("use_isotope_conditioning", True)
        self.use_fingerprint = m.get("use_fingerprint_conditioning", True)
        self.fingerprint_bits = m.get("fingerprint_bits", 4096)
        self.fingerprint_radius = m.get("fingerprint_radius", 2)
        # corrupt the true fingerprint during training, like a predicted one at inference
        self.fp_noise_prob = float(config.data.get("fingerprint_noise_prob", 0.0))
        lvl = config.data.get("fingerprint_noise_level", [0.10, 0.25])
        try:
            lvl = list(lvl)
            self.fp_noise_level = (float(lvl[0]), float(lvl[1])) if len(lvl) >= 2 else (0.0, float(lvl[0]))
        except TypeError:
            self.fp_noise_level = (0.0, float(lvl))
        self.exclude_inchi = self._load_exclude_set(config)
        if self.exclude_inchi:
            print(f"[MSBD3] Loaded {len(self.exclude_inchi)} molecules for exclusion filtering")

    def _load_exclude_set(self, config):
        exclude_file = config.data.get("exclude_inchikeys_file", None)
        if exclude_file and not os.path.exists(exclude_file):
            # training without the val/test exclusion list would leak benchmark molecules
            raise FileNotFoundError(f"data.exclude_inchikeys_file not found: {exclude_file}")
        if exclude_file:
            return frozenset(pd.read_csv(exclude_file)["inchi"].tolist())
        return None

    def _fingerprint(self, smiles):
        if smiles is None:
            return torch.zeros(self.fingerprint_bits, dtype=torch.float32), 0.0
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return torch.zeros(self.fingerprint_bits, dtype=torch.float32), 0.0
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=self.fingerprint_radius, nBits=self.fingerprint_bits)
            arr = np.zeros((self.fingerprint_bits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return torch.from_numpy(arr), 1.0
        except Exception:
            return torch.zeros(self.fingerprint_bits, dtype=torch.float32), 0.0

    def _corrupt_fp(self, fp):
        """Symmetric corruption: switch off a fraction of the on-bits and switch on as many off-bits."""
        if self.fp_noise_prob <= 0.0 or np.random.rand() > self.fp_noise_prob:
            return fp
        f = float(np.random.uniform(self.fp_noise_level[0], self.fp_noise_level[1]))
        on = torch.nonzero(fp > 0).flatten()
        if on.numel() == 0:
            return fp
        k = max(1, int(f * on.numel()))
        out = fp.clone()
        out[on[torch.randperm(on.numel())[:k]]] = 0.0
        off = torch.nonzero(fp == 0).flatten()
        add_k = min(k, off.numel())
        if add_k > 0:
            out[off[torch.randperm(off.numel())[:add_k]]] = 1.0
        return out

    def _safe_string(self, ex):
        """SAFE string of a corpus row (``input``, ``safe``, or converted from ``smiles``)."""
        if ex.get("input"):
            return ex["input"]
        if ex.get("safe"):
            return ex["safe"]
        smi = ex.get("smiles")
        if smi:
            try:
                s = smiles_to_safe(smi)
                if s:
                    return s
            except Exception:
                pass
        return ""

    def __call__(self, examples):
        for ex in examples:
            if not ex.get("input"):
                ex["input"] = self._safe_string(ex)

        if self.remove_stereo:
            for ex in examples:
                mol = Chem.MolFromSmiles(ex["input"])
                if mol is not None:
                    Chem.RemoveStereochemistry(mol)
                    smi = smiles_to_safe(Chem.MolToSmiles(mol))
                    if smi is not None:
                        ex["input"] = smi

        batch = self.tokenizer(
            [ex["input"] for ex in examples],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        batch.pop("token_type_ids", None)

        smiles_cache = [safe_to_smiles(ex["input"], fix=True) for ex in examples]

        exclude_mask = None
        if self.exclude_inchi is not None:
            exclude_mask = []
            for smi in smiles_cache:
                mol = Chem.MolFromSmiles(smi) if smi else None
                inchi = Chem.MolToInchiKey(mol) if mol else None
                excl = bool(inchi and inchi.split("-")[0] in self.exclude_inchi)
                exclude_mask.append(0.0 if excl else 1.0)

        if self.use_mass:
            masses = []
            for smi in smiles_cache:
                try:
                    masses.append(float(mass_from_smi(smi)) if smi else 0.0)
                except Exception:
                    masses.append(0.0)
            batch["mass"] = torch.tensor(masses, dtype=torch.float32)

        if self.use_isotope:
            iso = [isotope_features_from_smiles(smi) for smi in smiles_cache]
            batch["isotope"] = torch.stack(iso) if iso else torch.zeros(0, ISOTOPE_FEATURE_DIM)

        if self.use_fingerprint:
            fps, masks = [], []
            for i, smi in enumerate(smiles_cache):
                fp, ok = self._fingerprint(smi)
                if ok:
                    fp = self._corrupt_fp(fp)
                fps.append(fp)
                if exclude_mask is not None and exclude_mask[i] == 0.0:
                    masks.append(0.0)
                else:
                    masks.append(ok)
            batch["fingerprint"] = torch.stack(fps)
            batch["fingerprint_mask"] = torch.tensor(masks, dtype=torch.float32)

        if exclude_mask is not None:
            batch["exclude_mask"] = torch.tensor(exclude_mask, dtype=torch.float32)
        return batch


def get_msbd3_dataloader(config):
    """Streaming SAFE-GPT loader, or a local dataset when ``data.use_safe_hf`` is False."""
    if config.data.use_safe_hf:
        hf_cache_dir = config.data.hf_cache_dir
        if hf_cache_dir is not None:
            os.environ.setdefault("HF_DATASETS_CACHE", str(hf_cache_dir))
            os.environ.setdefault("HF_HUB_CACHE", str(hf_cache_dir))

        import datasets
        from datasets.distributed import split_dataset_by_node

        stream_ds = None
        try:
            from huggingface_hub import snapshot_download

            snapshot_dir = snapshot_download(
                "datamol-io/safe-gpt", repo_type="dataset", cache_dir=hf_cache_dir, local_files_only=True
            )
            train_files = sorted(glob.glob(os.path.join(snapshot_dir, "data/train/*.parquet")))
            if train_files:
                print(f"[MSBD3] Streaming {len(train_files)} local parquet files from {snapshot_dir}")
                stream_ds = datasets.load_dataset(
                    "parquet", data_files={"train": train_files}, streaming=True, split="train"
                )
        except Exception as e:
            print("[MSBD3] Local safe-gpt snapshot not found, falling back to Hub streaming:", e)

        if stream_ds is None:
            stream_ds = datasets.load_dataset(
                "datamol-io/safe-gpt", streaming=True, split="train",
                cache_dir=hf_cache_dir, download_mode="reuse_cache_if_exists",
            )

        rank, world_size = _rank_world()
        if world_size > 1:
            stream_ds = split_dataset_by_node(stream_ds, rank=rank, world_size=world_size)

        resume_step = config.loader.get("resume_step", 0) or 0
        num_workers = config.loader.num_workers
        persistent_workers = config.loader.get("persistent_workers", True)
        if resume_step > 0:
            skip_count = resume_step * config.loader.batch_size
            print(f"[MSBD3] Rank {rank}: skipping {skip_count} samples (resume step {resume_step})")
            stream_ds = stream_ds.skip(skip_count)
            if num_workers > 0:
                num_workers = 0
                persistent_workers = False
        if num_workers == 0:
            persistent_workers = False

        return torch.utils.data.DataLoader(
            stream_ds,
            batch_size=config.loader.batch_size,
            collate_fn=MSBD3Collator(config),
            num_workers=num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=False,
            persistent_workers=persistent_workers,
        )

    user_data_path = config.data.get("data_path", None)
    if not user_data_path:
        raise ValueError("config.data.data_path must be provided when data.use_safe_hf is False.")
    return torch.utils.data.DataLoader(
        UserDataset(user_data_path),
        batch_size=config.loader.batch_size,
        collate_fn=MSBD3Collator(config),
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.pin_memory,
        shuffle=True,
        persistent_workers=True,
    )
