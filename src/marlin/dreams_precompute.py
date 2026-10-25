"""Precompute DreaMS embeddings for SIRIUS ``.ms`` spectra or an MGF file (DreaMS environment).

The released NPLIB1 cache was made with:

    python -m marlin.dreams_precompute --spec-folder data/canopus/spec_files --emb-kind backbone \
        --finetuned-backbone weights/dreams_fp/backbone_finetuned_fp.pt \
        --out data/marlin/dreams_embeddings_nplib1.npz
"""

import argparse
import glob
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
from tqdm import tqdm


def parse_ms_file(path):
    """Parse a SIRIUS .ms file into ``(precursor_mz, peaks)``; ``(None, None)`` if either is missing."""
    prec_mz = None
    peaks = []
    in_peaks = False
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                key = line[1:].split(maxsplit=1)
                tag = key[0].lower()
                if tag == "parentmass" and len(key) > 1:
                    try:
                        prec_mz = float(key[1].split()[0])
                    except ValueError:
                        pass
                # peaks belong to any ms2/collision/energy block, never ms1
                in_peaks = tag.startswith("ms2") or tag.startswith("collision") or tag.startswith("energy")
                continue
            if line.startswith("#"):
                continue
            if in_peaks:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        pass
    if prec_mz is None or len(peaks) == 0:
        return None, None
    return prec_mz, np.asarray(peaks, dtype=np.float64)


def iter_mgf_blocks(path):
    """Yield ``(spec_id, precursor_mz, peaks)`` from an MGF file (id from ID= or TITLE=)."""
    spec_id, prec_mz, peaks = None, None, []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "BEGIN IONS":
                spec_id, prec_mz, peaks = None, None, []
                continue
            if line == "END IONS":
                if spec_id is not None and prec_mz is not None and peaks:
                    yield spec_id, prec_mz, np.asarray(peaks, dtype=np.float64)
                spec_id, prec_mz, peaks = None, None, []
                continue
            if "=" in line:  # header line
                k, _, v = line.partition("=")
                k = k.strip().upper()
                if k == "ID":
                    spec_id = v.strip()
                elif k == "TITLE" and spec_id is None:
                    spec_id = v.strip()
                elif k == "PARENTMASS" or (k == "PEPMASS" and prec_mz is None):
                    try:
                        prec_mz = float(v.split()[0])
                    except (ValueError, IndexError):
                        pass
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    peaks.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass


def _load_aug_smiles(labels_tsv):
    """Map spectrum id -> SMILES from a tab-separated labels file."""
    import csv
    m = {}
    with open(labels_tsv, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid, smi = row.get("spec"), row.get("smiles")
            if sid and smi:
                m[sid] = smi
    return m


_PROTON = 1.0072764665789
_PT = None


def _formula_monoiso(formula):
    """Monoisotopic mass of a chemical formula string (e.g. 'C12H19N5O2') via RDKit's table."""
    global _PT
    if _PT is None:
        from rdkit.Chem import GetPeriodicTable
        _PT = GetPeriodicTable()
    import re
    m = 0.0
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if el:
            m += _PT.GetMostCommonIsotopeMass(el) * (int(n) if n else 1)
    return m


def iter_subform_blocks(subform_dir):
    """Yield ``(spec_id, precursor_mz, peaks)`` from MIST subformula JSON files ([M+H]+ only)."""
    import glob as _glob
    import itertools as _it
    import json as _json
    paths = _it.chain(_glob.iglob(os.path.join(subform_dir, "*", "*.json")),
                      _glob.iglob(os.path.join(subform_dir, "*.json")))
    for path in paths:
        try:
            d = _json.load(open(path, encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if d.get("cand_ion") != "[M+H]+":
            continue
        ot = d.get("output_tbl") or {}
        mz, inten = ot.get("mz"), ot.get("ms2_inten")
        cf = d.get("cand_form")
        if not mz or not inten or len(mz) != len(inten) or not cf:
            continue
        prec_mz = _formula_monoiso(cf) + _PROTON
        peaks = np.asarray(list(zip(mz, inten)), dtype=np.float64)
        spec_id = os.path.splitext(os.path.basename(path))[0]
        yield spec_id, prec_mz, peaks


def main():
    ap = argparse.ArgumentParser(description="Precompute DreaMS embeddings for .ms spectra")
    ap.add_argument("--spec-folder", default="data/canopus/spec_files")
    ap.add_argument("--source", default="ms", choices=["ms", "mgf", "iceberg"],
                    help="ms = SIRIUS .ms folder (real spectra); mgf = one MGF file (--mgf; ids from ID=/TITLE=); "
                         "iceberg = an MGF of ICEBERG synthetic spectra (+ spec->smiles mapping, --aug-labels)")
    ap.add_argument("--mgf", default=None, help="(--source mgf|iceberg) the MGF file to stream")
    ap.add_argument("--aug-labels", default=None,
                    help="(--source iceberg) spec_id->smiles tsv; stored as aug_smiles + <out>.smiles.tsv")
    ap.add_argument("--aug-limit", type=int, default=None, help="(--source iceberg) only first N synthetic spectra")
    ap.add_argument("--out", default="data/dreams/dreams_emb_canopus.npz")
    ap.add_argument("--emb-kind", default="backbone", choices=["backbone", "contrastive", "all_peaks"],
                    help="backbone/contrastive = one vector per spectrum; all_peaks = ALL per-peak token "
                         "embeddings as a set (fp16) + mask (richest; for set-encoder head / adapter)")
    ap.add_argument("--n-highest-peaks", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--emb-ckpt", default=None)
    ap.add_argument("--ssl-ckpt", default=None)
    ap.add_argument("--finetuned-backbone", default=None,
                    help="backbone_finetuned_fp.pt from finetune_dreams_fp_backbone (use --emb-kind backbone)")
    ap.add_argument("--limit", type=int, default=None, help="debug: only first N files")
    args = ap.parse_args()
    if args.source in ("mgf", "iceberg") and (not args.mgf or (args.source == "iceberg" and not args.aug_labels)):
        ap.error("--source mgf needs --mgf; --source iceberg needs --mgf and --aug-labels")

    from marlin.dreams_adapter import DreamsEmbedder, DEFAULT_EMB_CKPT, DEFAULT_SSL_CKPT

    embedder = DreamsEmbedder(
        emb_ckpt=args.emb_ckpt or DEFAULT_EMB_CKPT,
        ssl_ckpt=args.ssl_ckpt or DEFAULT_SSL_CKPT,
        emb_kind=args.emb_kind,
        n_highest_peaks=args.n_highest_peaks,
        finetuned_backbone=args.finetuned_backbone,
    )
    print(f"[dreams-precompute] DreaMS loaded: d_model={embedder.d_model} emb_kind={args.emb_kind}")

    ids, specs, skipped = [], [], 0
    if args.source in ("mgf", "iceberg"):
        print(f"[dreams-precompute] streaming MGF {args.mgf}")
        for spec_id, prec_mz, peaks in tqdm(iter_mgf_blocks(args.mgf), desc="parsing mgf"):
            ids.append(spec_id)
            specs.append((peaks, prec_mz))
            if args.aug_limit and len(ids) >= args.aug_limit:
                break
        print(f"[dreams-precompute] parsed {len(ids)} synthetic spectra")
    else:
        files = sorted(glob.glob(os.path.join(args.spec_folder, "*.ms")))
        if args.limit:
            files = files[: args.limit]
        print(f"[dreams-precompute] {len(files)} .ms files in {args.spec_folder}")
        for fp in tqdm(files, desc="parsing .ms"):
            spec_id = os.path.splitext(os.path.basename(fp))[0]
            prec_mz, peaks = parse_ms_file(fp)
            if prec_mz is None:
                skipped += 1
                continue
            ids.append(spec_id)
            specs.append((peaks, prec_mz))
        print(f"[dreams-precompute] parsed {len(ids)} spectra, skipped {skipped}")

    embs, mask = embedder.embed(specs, batch_size=args.batch_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = dict(
        ids=np.array(ids),
        embs=embs if embs.dtype == np.float16 else embs.astype(np.float32),
        emb_kind=np.array(args.emb_kind),
        d_model=np.array(embedder.d_model),
        n_highest_peaks=np.array(args.n_highest_peaks),
    )
    if mask is not None:
        payload["mask"] = mask.astype(np.float32)
    if args.source == "iceberg":
        # SMILES of each synthetic spectrum (the head's training target)
        spec2smi = _load_aug_smiles(args.aug_labels)
        smis = [spec2smi.get(i, "") for i in ids]
        n_miss = sum(1 for s in smis if not s)
        payload["aug_smiles"] = np.array(smis)
        sib = args.out + ".smiles.tsv"
        with open(sib, "w") as f:
            f.write("spec\tsmiles\n")
            for i, s in zip(ids, smis):
                f.write(f"{i}\t{s}\n")
        print(f"[dreams-precompute] aug_smiles: {len(smis)-n_miss}/{len(smis)} mapped ({n_miss} missing) -> {sib}")
    np.savez_compressed(args.out, **payload)
    print(f"[dreams-precompute] wrote embs={embs.shape} mask={None if mask is None else mask.shape} -> {args.out}")


if __name__ == "__main__":
    main()
