"""Run MARLIN on an NPLIB1 split and score the predictions against the reference structures.

Writes predictions.jsonl, constraint_trace.jsonl, metrics.csv, aggregate_statistics.json,
config.yaml and mass_table.parquet to the output directory.
"""

import argparse
import os
import sys
import time
import random
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_PROJECT_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np
import torch
import yaml
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

from dlm.utils.benchmark_utils import (
    normalize_formula,
    compute_morgan_fingerprint,
    compute_tanimoto_similarity,
    get_inchikey_first_block,
    binarize_fingerprint,
)
from mist.utils.chem_utils import mass_from_smi
from mist.data import get_paired_spectra, get_paired_featurizer, SpectraMolDataset, PresetSpectraSplitter
from mist.data.datasets import get_paired_loader

from marlin.sampler import MassShellSampler
from marlin.encoders import build_encoder
from marlin.chem.mass_features import isotope_features_from_spectrum, neutral_mass_from_precursor
from marlin.chem.mass_table import build_mass_table, save_mass_table
from marlin.logging_utils import RunLogger, save_metrics, snapshot_config, aggregate_predictions


def parse_args():
    p = argparse.ArgumentParser(description="MARLIN inference / benchmark")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--mist-checkpoint", type=str, default=None)
    p.add_argument("--msbd3-checkpoint", "--decoder-checkpoint", dest="msbd3_checkpoint", type=str, default=None,
                   help="MARLIN block-diffusion decoder checkpoint (overrides msbd3_checkpoint in the config)")
    p.add_argument("--encoder", type=str, default=None, choices=["mist", "dreams"],
                   help="spectrum -> fingerprint encoder: mist | dreams (fine-tuned DreaMS + fp head)")
    p.add_argument("--dreams-embeddings", type=str, default=None,
                   help="DreaMS embedding .npz cache (encoder=dreams)")
    p.add_argument("--dreams-head", type=str, default=None,
                   help="trained DreaMS fingerprint-head checkpoint (encoder=dreams)")
    p.add_argument("--dreams-threshold", type=float, default=None,
                   help="override binarization threshold for the DreaMS head")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--split", type=str, default=None, choices=["val", "test"])
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=1,
                   help="split the split into N strided shards for parallel multi-GPU eval.")
    p.add_argument("--shard-idx", type=int, default=0, help="which shard in [0,num_shards) this process runs.")
    p.add_argument("--num-candidates", type=int, default=None)
    p.add_argument("--ppm-tol", type=float, default=None)
    p.add_argument("--mass-source", type=str, default=None, choices=["true_mass", "precursor"])
    p.add_argument("--adduct", type=str, default=None)
    p.add_argument("--use-isotope", action="store_true")
    p.add_argument("--isotope-source", type=str, default=None, choices=["measured", "none"],
                   help="measured = from the observed MS1 envelope around the precursor (default)")
    p.add_argument("--no-mass-shell-prune", action="store_true", help="ablation: disable the heavy-atom prune")
    p.add_argument("--no-mass-conditioning", action="store_true",
                   help="ablation: drop the mass conditioning token")
    p.add_argument("--h-slack", type=float, default=None, help="hydrogen slack of the EOS interval (default 4)")
    p.add_argument("--no-kv-cache", action="store_true", help="recompute the prefix instead of caching it (same output)")
    p.add_argument("--fp-diversity", type=float, default=None,
                   help="fraction of candidates decoded from a perturbed fingerprint (paper: 1.0)")
    p.add_argument("--fp-drop-rate", type=float, default=None,
                   help="fraction of on-bits dropped per perturbation (default 0.3)")
    p.add_argument("--softmax-temp", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_config(path, args):
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("mist_encoder", {})
    cfg.setdefault("encoder", {})  # encoder.type: mist (default) | dreams
    cfg["encoder"].setdefault("type", "mist")
    cfg.setdefault("data", {})
    cfg.setdefault("fingerprint", {"bits": 4096, "radius": 2, "threshold": 0.187})
    cfg.setdefault("inference", {})
    cfg.setdefault("output", {"results_dir": "results/msbd3"})
    inf = cfg["inference"]
    inf.setdefault("num_candidates", 64)
    inf.setdefault("ppm_tol", 10.0)
    inf.setdefault("mass_source", "true_mass")
    inf.setdefault("adduct", "[M+H]+")
    inf.setdefault("use_isotope", False)
    inf.setdefault("isotope_source", "measured")
    inf.setdefault("enable_mass_shell_prune", True)
    inf.setdefault("h_slack", 4.0)
    inf.setdefault("use_kv_cache", True)
    inf.setdefault("fp_diversity", 0.0)
    inf.setdefault("fp_drop_rate", 0.3)
    inf.setdefault("softmax_temp", 1.0)
    inf.setdefault("top_k", 10)
    inf.setdefault("split", "test")
    inf.setdefault("max_spectra", None)

    if args.encoder:
        cfg["encoder"]["type"] = args.encoder
    if args.dreams_embeddings:
        cfg["encoder"]["embeddings"] = args.dreams_embeddings
    if args.dreams_head:
        cfg["encoder"]["head_checkpoint"] = args.dreams_head
    if args.dreams_threshold is not None:
        cfg["encoder"]["fp_threshold"] = args.dreams_threshold
    if args.mist_checkpoint:
        cfg["mist_encoder"]["checkpoint"] = args.mist_checkpoint
    if args.msbd3_checkpoint:
        cfg["msbd3_checkpoint"] = args.msbd3_checkpoint
    if args.output_dir:
        cfg["output"]["results_dir"] = args.output_dir
    if args.split:
        inf["split"] = args.split
    if args.max_spectra is not None:
        inf["max_spectra"] = args.max_spectra
    if args.num_candidates is not None:
        inf["num_candidates"] = args.num_candidates
    if args.ppm_tol is not None:
        inf["ppm_tol"] = args.ppm_tol
    if args.mass_source:
        inf["mass_source"] = args.mass_source
    if args.adduct:
        inf["adduct"] = args.adduct
    if args.use_isotope:
        inf["use_isotope"] = True
    if args.isotope_source:
        inf["isotope_source"] = args.isotope_source
    if args.no_mass_shell_prune:
        inf["enable_mass_shell_prune"] = False
    if args.h_slack is not None:
        inf["h_slack"] = args.h_slack
    if args.no_kv_cache:
        inf["use_kv_cache"] = False
    if args.fp_diversity is not None:
        inf["fp_diversity"] = args.fp_diversity
    if args.fp_drop_rate is not None:
        inf["fp_drop_rate"] = args.fp_drop_rate
    if args.softmax_temp is not None:
        inf["softmax_temp"] = args.softmax_temp
    if args.top_k is not None:
        inf["top_k"] = args.top_k
    return cfg


def load_spec_data(data_cfg, enc_cfg, split):
    labels_file = data_cfg.get("labels_file", os.path.join(data_cfg["datadir"], "labels.tsv"))
    split_file = data_cfg.get("split_file", os.path.join(data_cfg["datadir"], "split.tsv"))
    spec_folder = data_cfg.get("spec_folder", os.path.join(data_cfg["datadir"], "spec_files"))
    subform_folder = data_cfg.get("subform_folder", os.path.join(data_cfg["datadir"], "default_subformulae"))

    spectra_list, mol_list = get_paired_spectra(
        labels_file=labels_file, spec_folder=spec_folder, prog_bars=True,
        split_file=split_file, split_val=split,
    )
    full = list(zip(spectra_list, mol_list))
    splitter = PresetSpectraSplitter(split_file=split_file)
    _, (train_d, val_d, test_d) = splitter.get_splits(full)
    split_data = {"train": train_d, "val": val_d, "test": test_d}[split]
    featurizer = get_paired_featurizer(
        spec_features="peakformula", mol_features="fingerprint", subform_folder=subform_folder,
        fp_names=["morgan4096"], magma_modulo=enc_cfg.get("magma_modulo", 2048),
        cls_type=enc_cfg.get("cls_type", "ms1"), inten_transform=enc_cfg.get("inten_transform", "float"),
        inten_prob=enc_cfg.get("inten_prob", 0.1), remove_prob=enc_cfg.get("remove_prob", 0.5),
    )
    return SpectraMolDataset(split_data, featurizer), split_data


def _target_mass(spec, true_smiles, source, adduct_default):
    if source == "precursor":
        try:
            spec.get_meta()  # triggers lazy load of parentmass / meta
            meta = spec.get_meta() or {}
            adduct = meta.get("ionization") or meta.get("adduct") or adduct_default
            charge = int(meta.get("charge", 1) or 1)
            M = neutral_mass_from_precursor(spec.parentmass, adduct, charge)
            if M is not None and M > 0:
                return float(M)
        except Exception:
            pass
        # fall through to true mass if precursor unavailable
    return float(mass_from_smi(true_smiles))


def _isotope_features(spec, inf):
    """Isotope features from the MS1 envelope around the precursor, or ``None``."""
    if not inf.get("use_isotope"):
        return None
    src = inf.get("isotope_source", "measured")
    if src == "none":
        return None
    try:
        spec.get_meta()  # ensure lazy load (sets parentmass)
        specs = spec.get_spec()
        arrays = [np.asarray(s) for s in specs if s is not None and len(s)] if specs else []
        peaks = np.concatenate(arrays, axis=0) if arrays else None
        return isotope_features_from_spectrum(peaks, getattr(spec, "parentmass", None))
    except Exception:
        return None


def main():
    args = parse_args()
    cfg = load_config(args.config, args)
    inf = cfg["inference"]
    fp_cfg = cfg["fingerprint"]
    out_dir = cfg["output"]["results_dir"]
    sharded = args.num_shards and args.num_shards > 1
    if sharded:
        out_dir = os.path.join(out_dir, f"shard{args.shard_idx}of{args.num_shards}")
    os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, split_data = load_spec_data(cfg["data"], cfg["mist_encoder"], inf["split"])
    encoder = build_encoder(cfg, device)
    sampler = MassShellSampler(cfg["msbd3_checkpoint"], device=device)
    if getattr(args, "no_mass_conditioning", False):
        sampler.model.use_mass = False   # ablation: drop the mass conditioning token
        print("[infer] ablation: mass conditioning token dropped")

    snapshot_config(out_dir, cfg)
    save_mass_table(build_mass_table(sampler.tokenizer), os.path.join(out_dir, "mass_table.parquet"), sampler.tokenizer)
    logger = RunLogger(out_dir, trace_every=cfg.get("trace_every", 200))

    fp_bits, fp_radius, fp_thr = fp_cfg["bits"], fp_cfg["radius"], fp_cfg["threshold"]
    top_k = inf["top_k"]
    dataloader = get_paired_loader(dataset, shuffle=False, batch_size=1, num_workers=0)
    max_spectra = inf["max_spectra"]
    predictions = []
    t0 = time.time()

    try:
        n_total = len(dataloader)
    except TypeError:
        n_total = None
    if max_spectra:
        n_total = min(max_spectra, n_total) if n_total else max_spectra

    for idx, batch in enumerate(tqdm(dataloader, total=n_total, desc="MS-BD3 inference")):
        if max_spectra and idx >= max_spectra:
            break
        if sharded and (idx % args.num_shards) != args.shard_idx:
            continue   # strided sharding
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        spec, mol = split_data[idx]
        true_smiles = mol.get_smiles()
        true_mol = Chem.MolFromSmiles(true_smiles)
        if true_mol is None:
            continue
        true_inchi = get_inchikey_first_block(mol.get_inchikey())
        true_formula = normalize_formula(rdMolDescriptors.CalcMolFormula(true_mol))
        true_fp = compute_morgan_fingerprint(true_smiles, fp_bits, fp_radius)

        M = _target_mass(spec, true_smiles, inf["mass_source"], inf["adduct"])
        iso = _isotope_features(spec, inf)

        common = dict(
            target_mass=M, isotope=iso, num_candidates=inf["num_candidates"], ppm_tol=inf["ppm_tol"],
            softmax_temp=inf["softmax_temp"], enable_mass_shell_prune=inf["enable_mass_shell_prune"],
            h_slack=inf["h_slack"], use_kv_cache=inf["use_kv_cache"], collect_trace=logger.should_trace(),
        )

        pred_fp_probs = encoder.predict_proba(spec=spec, batch=batch)
        pred_fp = binarize_fingerprint(pred_fp_probs, encoder.threshold)
        mist_tani = float(compute_tanimoto_similarity(true_fp, pred_fp))
        result = sampler.generate(
            fingerprint=pred_fp, fp_diversity=inf["fp_diversity"], fp_drop_rate=inf["fp_drop_rate"], **common,
        )
        ranked = sampler.rank(result["candidates"], pred_fp, fp_bits=fp_bits, fp_radius=fp_radius)

        # annotate candidates with ground-truth comparisons
        for c in ranked[:max(top_k, 10)]:
            cand_fp = compute_morgan_fingerprint(c["smiles"], fp_bits, fp_radius)
            c["tanimoto_to_true"] = float(compute_tanimoto_similarity(true_fp, cand_fp)) if cand_fp is not None else 0.0
            try:
                cm = Chem.MolFromSmiles(c["smiles"])
                c["formula"] = normalize_formula(rdMolDescriptors.CalcMolFormula(cm)) if cm else ""
                c["formula_match"] = bool(c["formula"] == true_formula)
                c["exact_match"] = bool(get_inchikey_first_block(Chem.MolToInchiKey(cm)) == true_inchi) if cm else False
            except Exception:
                c["formula"], c["formula_match"], c["exact_match"] = "", False, False

        top = ranked[:top_k] if top_k else ranked
        # recall_any: the true molecule anywhere among the candidates
        _ann = max(top_k or 0, 10)
        recall_any = any(c.get("exact_match") for c in ranked[:_ann])
        if not recall_any:
            for c in ranked[_ann:]:
                cm = Chem.MolFromSmiles(c["smiles"])
                if cm and get_inchikey_first_block(Chem.MolToInchiKey(cm)) == true_inchi:
                    recall_any = True
                    break
        record = {
            "spectrum_id": spec.get_spec_name(),
            "target_mass": M,
            "mass_source": inf["mass_source"],
            "true_smiles": true_smiles,
            "true_formula": true_formula,
            "mist_tanimoto": mist_tani,
            "candidates": [
                {k: c.get(k) for k in (
                    "rank", "smiles", "safe", "formula", "mass_pred", "mass_err_ppm",
                    "accepted", "formula_match", "exact_match", "tanimoto_to_f",
                    "tanimoto_to_true", "decoder_nll", "n_blocks",
                )}
                for c in top
            ],
            "top1_exact": bool(top and top[0].get("exact_match")),
            "top10_exact": bool(any(c.get("exact_match") for c in ranked[:10])),
            "recall_any": bool(recall_any),
            "deadend": bool(any(c.get("deadend") for c in result["candidates"])),
            "eos_forced": bool(any(t.get("eos_forced") for t in result["trace"])),
        }
        logger.log_prediction(record)
        if result["trace"]:
            logger.log_trace({"spectrum_id": spec.get_spec_name(), "target_mass": M, "trace": result["trace"]})
        predictions.append(record)

    logger.close()
    elapsed = time.time() - t0
    metrics = aggregate_predictions(predictions)
    metrics["latency_sec_per_spectrum"] = elapsed / max(len(predictions), 1)
    metrics["block_size"] = sampler.block_size
    metrics["ppm_tol"] = inf["ppm_tol"]
    metrics["enable_mass_shell_prune"] = inf["enable_mass_shell_prune"]
    metrics["mass_source"] = inf["mass_source"]
    off = [name for name, on in (("prune", inf["enable_mass_shell_prune"]),
                                 ("mass-conditioning", not getattr(args, "no_mass_conditioning", False)),
                                 ("diversity", bool(inf["fp_diversity"]))) if not on]
    cond = "full" if not off else "no-" + "+no-".join(off)
    save_metrics(out_dir, metrics, dataset=os.path.basename(cfg["data"].get("datadir", "")), condition=cond)
    if sharded:
        open(os.path.join(out_dir, ".done"), "w").close()   # marks the shard as finished

    print("\n==== MARLIN inference summary ====")
    for k in ("total_spectra", "top1_acc", "top10_acc", "recall_any",
              "tanimoto_top1_mean", "tanimoto_top10_mean", "mass_match_rate",
              "formula_recovery@1", "formula_recovery@10", "deadend_rate",
              "latency_sec_per_spectrum"):
        print(f"  {k}: {metrics.get(k)}")
    print(f"Results written to: {out_dir}")


if __name__ == "__main__":
    main()
