#!/usr/bin/env python3
"""Merge sharded runs (``--num-shards N``) into ``predictions_merged.jsonl`` and ``metrics_merged.json``.

  python -m marlin.merge_shards results/marlin_dreams
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from marlin.logging_utils import aggregate_predictions  # noqa: E402


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    base = sys.argv[1]
    shard_files = sorted(glob.glob(os.path.join(base, "shard*of*", "predictions.jsonl")))
    if not shard_files:
        sys.exit(f"No shard predictions found under {base}/shard*of*/predictions.jsonl")

    done = sorted(glob.glob(os.path.join(base, "shard*of*", ".done")))
    print(f"shards with predictions: {len(shard_files)} | shards marked .done: {len(done)}")

    preds, seen = [], set()
    for f in shard_files:
        n = 0
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r.get("spectrum_id")
            if sid in seen:        # guard against overlap on re-runs
                continue
            seen.add(sid); preds.append(r); n += 1
        print(f"  {os.path.relpath(f, base)}: {n} records")

    merged_path = os.path.join(base, "predictions_merged.jsonl")
    with open(merged_path, "w") as fh:
        for r in preds:
            fh.write(json.dumps(r) + "\n")

    m = aggregate_predictions(preds)
    print("\n==== merged metrics ====")
    for k in ("total_spectra", "top1_acc", "top10_acc", "recall_any",
              "tanimoto_top1_mean", "tanimoto_top10_mean", "formula_recovery@1",
              "formula_recovery@10", "mass_match_rate", "deadend_rate"):
        print(f"  {k}: {m.get(k)}")
    with open(os.path.join(base, "metrics_merged.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    print(f"\nmerged predictions -> {merged_path}")
    print(f"merged metrics     -> {os.path.join(base, 'metrics_merged.json')}")


if __name__ == "__main__":
    main()
