#!/usr/bin/env python
"""Top-1 / Top-10 MCES (myopic maximum common edge subgraph distance) for a predictions.jsonl.

Writes means over all spectra (a missing or invalid candidate counts as twice the bond count of
the true molecule) and over spectra with at least one candidate.

  python -m marlin.compute_mces --predictions results/marlin_dreams/predictions.jsonl --workers 16 --threads 1
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from rdkit import Chem, RDLogger
from myopic_mces import MCES

RDLogger.DisableLog("rdApp.*")

_THRESHOLD = 100.0  # MCES search threshold (well above observed distances)


def _num_bonds(smi):
    m = Chem.MolFromSmiles(smi)
    return (m, m.GetNumBonds()) if m is not None else (None, 0)


def _mces_pair(true_smi_canon, true_nb, pred_smi, current_best, solver, solver_options):
    """MCES between the true molecule and one candidate; ``current_best`` is the best so far."""
    pmol = Chem.MolFromSmiles(pred_smi) if pred_smi else None
    if pmol is None:
        return 2.0 * true_nb  # invalid candidate -> pessimistic fallback
    pnb = pmol.GetNumBonds()
    if current_best <= 0.0:
        return current_best
    pmol_smi = Chem.MolToSmiles(pmol)
    if pmol_smi == true_smi_canon:
        return 0.0
    lb = float(abs(true_nb - pnb))
    if lb >= current_best:
        return lb  # lower bound already not better -> skip the ILP
    eff_thr = _THRESHOLD if _THRESHOLD == -1.0 else min(_THRESHOLD, current_best)
    try:
        val = float(
            MCES(
                true_smi_canon,
                pmol_smi,
                solver=solver,
                threshold=eff_thr,
                always_stronger_bound=False,
                solver_options=solver_options,
                catch_errors=True,
            )[1]
        )
        if val < 0:
            val = float(true_nb + pnb)
    except Exception:
        val = float(true_nb + pnb)
    return val


def _spectrum_mces(rec, top_k, solver, solver_options):
    """Return (mces@1, mces@k) for one spectrum record."""
    true_smi = rec.get("true_smiles")
    tmol, true_nb = _num_bonds(true_smi) if true_smi else (None, 0)
    fallback = 2.0 * true_nb if tmol is not None else float(_THRESHOLD)
    if tmol is None:
        return fallback, fallback
    true_canon = Chem.MolToSmiles(tmol)

    cands = sorted(rec.get("candidates", []), key=lambda c: c.get("rank", 1e9))
    best = fallback
    mces_at1 = fallback
    for i, c in enumerate(cands[:top_k]):
        v = _mces_pair(true_canon, true_nb, c.get("smiles"), best, solver, solver_options)
        best = min(best, v)
        if i == 0:
            mces_at1 = best
        if best <= 0.0:
            break
    return mces_at1, best


def _worker(args):
    rec, top_k, solver, threads, time_limit = args
    opts = {"msg": False}
    if threads:
        opts["threads"] = int(threads)
    if time_limit:
        opts["timeLimit"] = float(time_limit)
    a, b = _spectrum_mces(rec, top_k, solver, opts)
    return rec.get("spectrum_id"), a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", default=None, help="output json (default: <pred_dir>/mces.json)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--solver", default="HiGHS",
                    help="HiGHS (open source; used for the paper) or GUROBI (needs a license; same exact MCES)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1, help="GUROBI threads per solve")
    ap.add_argument("--time-limit", type=float, default=300.0, help="per-solve seconds")
    ap.add_argument("--max-spectra", type=int, default=0, help="0 = all (debug)")
    ap.add_argument("--cache", default=None,
                    help="resume cache jsonl (default: <pred_dir>/mces_cache_k{top_k}.jsonl)")
    ap.add_argument("--no-cache", action="store_true", help="disable the stop/resume cache")
    args = ap.parse_args()

    recs = []
    with open(args.predictions) as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if args.max_spectra:
        recs = recs[: args.max_spectra]

    pred_dir = os.path.dirname(args.predictions)
    out_path = args.out or os.path.join(pred_dir, "mces.json")
    cache_path = None if args.no_cache else (
        args.cache or os.path.join(pred_dir, f"mces_cache_k{args.top_k}.jsonl"))

    # Resume: load any spectra already solved (keyed by spectrum_id).
    done = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    done[d["spectrum_id"]] = (d["mces_top1"], d["mces_topk"])
                except Exception:
                    pass
        print(f"[resume] {len(done)} spectra already cached in {cache_path}")

    todo = [r for r in recs if r.get("spectrum_id") not in done]
    print(f"[plan] total={len(recs)} cached={len(recs) - len(todo)} todo={len(todo)} "
          f"solver={args.solver} workers={args.workers}")

    tasks = [(r, args.top_k, args.solver, args.threads, args.time_limit) for r in todo]
    cfh = open(cache_path, "a") if cache_path else None

    def _record(sid, a, b):
        done[sid] = (a, b)
        if cfh is not None:
            cfh.write(json.dumps({"spectrum_id": sid, "mces_top1": a, "mces_topk": b}) + "\n")
            cfh.flush()
            os.fsync(cfh.fileno())

    try:
        if args.workers <= 1:
            for i, t in enumerate(tasks, 1):
                sid, a, b = _worker(t)
                _record(sid, a, b)
                if i % 50 == 0:
                    print(f"[progress] {i}/{len(tasks)} this run", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_worker, t) for t in tasks]
                for i, fut in enumerate(as_completed(futs), 1):
                    sid, a, b = fut.result()
                    _record(sid, a, b)
                    if i % 50 == 0:
                        print(f"[progress] {i}/{len(futs)} this run", flush=True)
    finally:
        if cfh is not None:
            cfh.close()

    vals1 = [done[r["spectrum_id"]][0] for r in recs if r.get("spectrum_id") in done]
    valsk = [done[r["spectrum_id"]][1] for r in recs if r.get("spectrum_id") in done]
    n = len(vals1)
    ans = [r for r in recs if r.get("spectrum_id") in done and r.get("candidates")]
    a1 = [done[r["spectrum_id"]][0] for r in ans]
    ak = [done[r["spectrum_id"]][1] for r in ans]
    out = {
        "predictions": os.path.abspath(args.predictions),
        "n_spectra": n,
        "n_total": len(recs),
        "n_answered": len(ans),
        "top_k": args.top_k,
        "mces_top1_mean": (sum(vals1) / n) if n else None,
        f"mces_top{args.top_k}_mean": (sum(valsk) / n) if n else None,
        "mces_top1_mean_answered": (sum(a1) / len(a1)) if a1 else None,
        f"mces_top{args.top_k}_mean_answered": (sum(ak) / len(ak)) if ak else None,
        "solver": args.solver,
        "threshold": _THRESHOLD,
        "cache": cache_path,
    }
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print(f"[written] {out_path}  (n={n}/{len(recs)})")


if __name__ == "__main__":
    main()
