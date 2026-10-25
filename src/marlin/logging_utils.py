"""Per-run output files: predictions, constraint traces, metrics and the config snapshot."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional


class JsonlWriter:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._f = open(path, "w")

    def write(self, record: Dict):
        self._f.write(json.dumps(record, default=_json_default) + "\n")
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


class RunLogger:
    """Writes ``predictions.jsonl`` and ``constraint_trace.jsonl`` for one run."""

    def __init__(self, out_dir: str, trace_every: int = 200):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.predictions = JsonlWriter(os.path.join(out_dir, "predictions.jsonl"))
        self.traces = JsonlWriter(os.path.join(out_dir, "constraint_trace.jsonl"))
        self.trace_every = trace_every
        self._n = 0

    def log_prediction(self, record: Dict):
        self.predictions.write(record)
        self._n += 1

    def should_trace(self) -> bool:
        return (self._n % max(self.trace_every, 1)) == 0

    def log_trace(self, record: Dict):
        self.traces.write(record)

    def close(self):
        self.predictions.close()
        self.traces.close()


def snapshot_config(out_dir: str, config) -> None:
    os.makedirs(out_dir, exist_ok=True)
    try:
        from omegaconf import OmegaConf

        text = OmegaConf.to_yaml(config) if not isinstance(config, dict) else None
    except Exception:
        text = None
    path = os.path.join(out_dir, "config.yaml")
    if text is not None:
        with open(path, "w") as f:
            f.write(text)
    else:
        import yaml

        with open(path, "w") as f:
            yaml.safe_dump(config, f, default_flow_style=False)


def save_metrics(out_dir: str, metrics: Dict, dataset: str = "", condition: str = "") -> None:
    """Write a single-row ``metrics.csv`` and ``aggregate_statistics.json``."""
    import pandas as pd

    os.makedirs(out_dir, exist_ok=True)
    row = {"dataset": dataset, "condition": condition, **metrics}
    pd.DataFrame([row]).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    with open(os.path.join(out_dir, "aggregate_statistics.json"), "w") as f:
        json.dump(row, f, indent=2, default=_json_default)


def aggregate_predictions(predictions: List[Dict], top_ks=(1, 10)) -> Dict:
    """Aggregate per-spectrum prediction records into run-level metrics."""
    n = len(predictions)
    if n == 0:
        return {"total_spectra": 0}

    def frac(pred_key):
        return sum(1.0 for p in predictions if p.get(pred_key)) / n

    out: Dict = {"total_spectra": n}
    for k in top_ks:
        out[f"top{k}_acc"] = frac(f"top{k}_exact")
    out["recall_any"] = frac("recall_any")   # true molecule anywhere among the candidates

    # fraction of candidates within the ppm tolerance
    n_cand = 0
    n_mass_ok = 0
    formula_rec = {k: 0.0 for k in top_ks}
    tani_top1 = []
    tani_top10 = []
    deadends = 0
    eos_forced = 0
    for p in predictions:
        cands = p.get("candidates", [])
        for c in cands:
            n_cand += 1
            if c.get("accepted"):
                n_mass_ok += 1
        for k in top_ks:
            topk = cands[:k]
            if any(c.get("formula_match") for c in topk):
                formula_rec[k] += 1.0
        if cands:
            tani_top1.append(cands[0].get("tanimoto_to_true", 0.0))
            tani_top10.append(max((c.get("tanimoto_to_true", 0.0) for c in cands[:10]), default=0.0))
        if p.get("deadend"):
            deadends += 1
        if p.get("eos_forced"):
            eos_forced += 1

    out["mass_match_rate"] = (n_mass_ok / n_cand) if n_cand else 0.0
    for k in top_ks:
        out[f"formula_recovery@{k}"] = formula_rec[k] / n
    out["tanimoto_top1_mean"] = (sum(tani_top1) / len(tani_top1)) if tani_top1 else 0.0
    out["tanimoto_top10_mean"] = (sum(tani_top10) / len(tani_top10)) if tani_top10 else 0.0
    out["deadend_rate"] = deadends / n
    out["eos_forced_rate"] = eos_forced / n
    return out
