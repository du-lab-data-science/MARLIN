#!/usr/bin/env python3
"""Summarize results directories: accuracy, MCES, Tanimoto, formula recovery, accuracy by mass.

Accuracy and formula recovery are over all spectra; MCES (from ``marlin.compute_mces``) and
Tanimoto are over spectra with at least one candidate.

  python -m marlin.summarize results/marlin_dreams results/marlin_mist
"""
import gzip
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from marlin.logging_utils import aggregate_predictions  # noqa: E402

MASS_BINS = ((0.0, 300.0, "<300"), (300.0, 500.0, "300-500"), (500.0, float("inf"), ">=500"))


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_records(d):
    for name in ("predictions_merged.jsonl", "predictions.jsonl", "predictions.jsonl.gz"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            with _open(p) as fh:
                return [json.loads(line) for line in fh if line.strip()], p
    raise FileNotFoundError(f"no predictions_merged.jsonl / predictions.jsonl(.gz) in {d}")


def summarize(d):
    recs, path = load_records(d)
    m = aggregate_predictions(recs)
    answered = {r["spectrum_id"] for r in recs if r.get("candidates")}
    out = {
        "dir": d, "predictions": os.path.basename(path), "n": len(recs), "answered": len(answered),
        "top1_acc": 100 * m["top1_acc"], "top10_acc": 100 * m["top10_acc"],
        "tani1": m["tanimoto_top1_mean"], "tani10": m["tanimoto_top10_mean"],
        "formula_recovery@1": 100 * m["formula_recovery@1"],
    }
    cache = os.path.join(d, "mces_cache_k10.jsonl")
    if not os.path.exists(cache) and os.path.exists(cache + ".gz"):
        cache += ".gz"
    if os.path.exists(cache):
        mces = {}
        with _open(cache) as fh:
            for line in fh:
                if line.strip():
                    c = json.loads(line)
                    mces[c["spectrum_id"]] = (c["mces_top1"], c["mces_topk"])
        vals = [mces[s] for s in answered if s in mces]
        if len(vals) < len(answered):
            print(f"  [warn] {d}: MCES cached for {len(vals)}/{len(answered)} answered spectra")
        if vals:
            out["mces1"] = statistics.mean(v[0] for v in vals)
            out["mces10"] = statistics.mean(v[1] for v in vals)
    bins = []
    for lo, hi, name in MASS_BINS:
        sel = [r for r in recs if lo <= r["target_mass"] < hi]
        acc = 100 * sum(1 for r in sel if r.get("top1_exact")) / len(sel) if sel else float("nan")
        bins.append((name, len(sel), acc))
    out["mass_bins"] = bins
    return out


def main():
    dirs = sys.argv[1:]
    if not dirs:
        sys.exit(__doc__)
    for d in dirs:
        s = summarize(d)
        mces1 = f"{s['mces1']:.2f}" if "mces1" in s else "n/a"
        mces10 = f"{s['mces10']:.2f}" if "mces10" in s else "n/a"
        print(f"\n== {d}  ({s['predictions']}: {s['n']} spectra, {s['answered']} with a candidate)")
        print(f"   Top-1 : acc {s['top1_acc']:.2f}%  MCES {mces1}  Tanimoto {s['tani1']:.2f}")
        print(f"   Top-10: acc {s['top10_acc']:.2f}%  MCES {mces10}  Tanimoto {s['tani10']:.2f}")
        print(f"   formula recovery@1: {s['formula_recovery@1']:.1f}% of all spectra")
        print("   Top-1 acc by mass: " + ", ".join(f"{n} Da {a:.1f}% (n={k})" for n, k, a in s["mass_bins"]))


if __name__ == "__main__":
    main()
