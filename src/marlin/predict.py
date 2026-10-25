"""Predict structures for your own MS/MS spectra (DreaMS encoder, no formula needed).

  1. DreaMS environment: embed the spectra with the fine-tuned backbone
       python -m marlin.dreams_precompute --spec-folder <dir of SIRIUS .ms files> \\
           --emb-kind backbone --finetuned-backbone weights/dreams_fp/backbone_finetuned_fp.pt --out emb.npz
     (or ``--source mgf --mgf spectra.mgf``)
  2. Main environment:
       python -m marlin.predict --spectra <same .ms dir or .mgf> --embeddings emb.npz --out results/mine

The neutral mass comes from each spectrum's precursor m/z and adduct (``--adduct`` when none is given).
Writes ``predictions.jsonl`` and ``predictions.tsv``.
"""

import argparse
import glob
import json
import os
import random
import sys
import time
from argparse import Namespace

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import rdMolDescriptors  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from dlm.utils.benchmark_utils import binarize_fingerprint  # noqa: E402
from marlin.chem.mass_features import neutral_mass_from_precursor  # noqa: E402
from marlin.encoders import build_encoder  # noqa: E402
from marlin.infer import load_config  # noqa: E402
from marlin.sampler import MassShellSampler  # noqa: E402


class _SpecId:
    """The encoder looks embeddings up by ``spec.get_spec_name()``."""

    def __init__(self, name):
        self._name = name

    def get_spec_name(self):
        return self._name


def _read_ms_meta(path):
    meta = {}
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(">"):
                key = line[1:].split(maxsplit=1)
                if len(key) == 2:
                    meta[key[0].lower()] = key[1].strip()
    return meta


def read_precursors(spectra, default_adduct):
    """{spec_id: (precursor_mz, adduct, charge)} from a folder of SIRIUS .ms files or one MGF file."""
    out = {}
    if os.path.isdir(spectra):
        for fp in sorted(glob.glob(os.path.join(spectra, "*.ms"))):
            meta = _read_ms_meta(fp)
            if "parentmass" not in meta:
                continue
            charge = meta.get("charge", "1").rstrip("+-") or "1"
            out[os.path.splitext(os.path.basename(fp))[0]] = (
                float(meta["parentmass"].split()[0]), meta.get("ionization", default_adduct), int(charge))
        return out
    block = None
    with open(spectra, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line == "BEGIN IONS":
                block = {}
            elif line == "END IONS" and block is not None:
                sid = block.get("ID") or block.get("TITLE")
                mz = block.get("PARENTMASS") or block.get("PEPMASS")
                if sid and mz:
                    adduct = block.get("ADDUCT") or block.get("IONIZATION") or block.get("PRECURSORTYPE") \
                        or default_adduct
                    charge = (block.get("CHARGE", "1").rstrip("+-") or "1")
                    out[sid] = (float(mz.split()[0]), adduct, int(charge))
                block = None
            elif block is not None and "=" in line:
                k, _, v = line.partition("=")
                block.setdefault(k.strip().upper(), v.strip())
    return out


def parse_args():
    p = argparse.ArgumentParser(description="MARLIN structure prediction for unlabeled spectra (DreaMS encoder)")
    p.add_argument("--spectra", required=True, help="folder of SIRIUS .ms files, or one .mgf file")
    p.add_argument("--embeddings", required=True, help="DreaMS embedding cache from marlin.dreams_precompute")
    p.add_argument("--config", default="src/marlin/configs/infer_nplib1_dreams.yaml")
    p.add_argument("--decoder-checkpoint", default=None, help="default: msbd3_checkpoint in the config")
    p.add_argument("--dreams-head", default=None, help="default: encoder.head_checkpoint in the config")
    p.add_argument("--adduct", default="[M+H]+", help="used when a spectrum does not state its adduct")
    p.add_argument("--num-candidates", type=int, default=None, help="decodes per spectrum (paper: 384)")
    p.add_argument("--top-k", type=int, default=10, help="candidates written per spectrum")
    p.add_argument("--ppm-tol", type=float, default=None)
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="output directory")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, Namespace(
        config=args.config, mist_checkpoint=None, msbd3_checkpoint=args.decoder_checkpoint, output_dir=args.out,
        split=None, max_spectra=args.max_spectra, num_candidates=args.num_candidates, ppm_tol=args.ppm_tol,
        mass_source="precursor", adduct=args.adduct, use_isotope=False, isotope_source=None,
        no_mass_shell_prune=False, h_slack=None, no_kv_cache=False, fp_diversity=None, fp_drop_rate=None,
        softmax_temp=None, top_k=args.top_k, seed=args.seed, encoder="dreams", dreams_embeddings=args.embeddings,
        dreams_head=args.dreams_head, dreams_threshold=None))
    inf, fp_cfg = cfg["inference"], cfg["fingerprint"]
    os.makedirs(args.out, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    precursors = read_precursors(args.spectra, args.adduct)
    encoder = build_encoder(cfg, device)
    ids = [s for s in precursors if s in encoder.cache]
    missing = len(precursors) - len(ids)
    if missing:
        print(f"[predict] {missing} spectra have no embedding in {args.embeddings} and are skipped")
    if args.max_spectra:
        ids = ids[: args.max_spectra]
    sampler = MassShellSampler(cfg["msbd3_checkpoint"], device=device)

    jsonl = open(os.path.join(args.out, "predictions.jsonl"), "w")
    tsv = open(os.path.join(args.out, "predictions.tsv"), "w")
    tsv.write("spectrum_id\trank\tsmiles\tformula\tmass_err_ppm\ttanimoto_to_predicted_fp\n")
    t0 = time.time()
    for i, sid in enumerate(ids, 1):
        mz, adduct, charge = precursors[sid]
        M = neutral_mass_from_precursor(mz, adduct, charge)
        if M is None:
            print(f"[predict] {sid}: unknown adduct {adduct!r}, skipped")
            continue
        fp = binarize_fingerprint(encoder.predict_proba(spec=_SpecId(sid)), encoder.threshold)
        result = sampler.generate(
            target_mass=M, isotope=None, fingerprint=fp, num_candidates=inf["num_candidates"],
            ppm_tol=inf["ppm_tol"], softmax_temp=inf["softmax_temp"],
            enable_mass_shell_prune=inf["enable_mass_shell_prune"], h_slack=inf["h_slack"],
            use_kv_cache=inf["use_kv_cache"], fp_diversity=inf["fp_diversity"], fp_drop_rate=inf["fp_drop_rate"],
        )
        ranked = sampler.rank(result["candidates"], fp, fp_bits=fp_cfg["bits"], fp_radius=fp_cfg["radius"])
        cands = []
        for rank, c in enumerate(ranked[: args.top_k], 1):
            mol = Chem.MolFromSmiles(c["smiles"])
            formula = rdMolDescriptors.CalcMolFormula(mol) if mol is not None else ""
            cands.append({"rank": rank, "smiles": c["smiles"], "formula": formula,
                          "mass_pred": c.get("mass_pred"), "mass_err_ppm": c.get("mass_err_ppm"),
                          "tanimoto_to_predicted_fp": c.get("tanimoto_to_f")})
            err = c.get("mass_err_ppm")
            tsv.write(f"{sid}\t{rank}\t{c['smiles']}\t{formula}\t{'' if err is None else f'{err:.2f}'}\t"
                      f"{c.get('tanimoto_to_f', 0.0):.4f}\n")
        jsonl.write(json.dumps({"spectrum_id": sid, "precursor_mz": mz, "adduct": adduct, "charge": charge,
                                "neutral_mass": M, "n_accepted": len(ranked), "candidates": cands}) + "\n")
        jsonl.flush(); tsv.flush()
        print(f"[predict] {i}/{len(ids)} {sid}: M={M:.4f} accepted={len(ranked)} "
              f"top1={cands[0]['smiles'] if cands else '-'}", flush=True)
    jsonl.close(); tsv.close()
    print(f"[predict] {len(ids)} spectra in {time.time() - t0:.0f}s -> {args.out}/predictions.{{jsonl,tsv}}")


if __name__ == "__main__":
    main()
