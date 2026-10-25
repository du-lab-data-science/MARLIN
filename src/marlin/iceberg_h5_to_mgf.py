"""Convert ms-pred ICEBERG ``preds.hdf5`` (raw predicted peaks) to an MGF plus a spec -> SMILES table.

  python -m marlin.iceberg_h5_to_mgf \
      --preds <save-dir>/preds.hdf5 --labels <ms-pred labels.tsv> --out data/dreams/iceberg_aug.mgf
"""

import argparse
import re

import numpy as np
import h5py
import pandas as pd

_PAT = re.compile(r"pred_([^/]+)/ikey [^/]+/collision ([^/]+)/spec$")


def main():
    ap = argparse.ArgumentParser(description="ICEBERG preds.hdf5 -> MGF (+ .labels.tsv) for DreaMS")
    ap.add_argument("--preds", required=True, help="ms-pred preds.hdf5 (sparse, non-binned)")
    ap.add_argument("--labels", required=True, help="ms-pred labels.tsv (spec->smiles, precursor)")
    ap.add_argument("--out", required=True, help="output .mgf (also writes <out>.labels.tsv)")
    ap.add_argument("--min-inten", type=float, default=1e-4, help="drop peaks below this (relative) intensity")
    args = ap.parse_args()

    df = pd.read_csv(args.labels, sep="\t")
    smi_by = dict(zip(df["spec"].astype(str), df["smiles"].astype(str)))
    prec_by = dict(zip(df["spec"].astype(str), df["precursor"].astype(float)))

    import glob as _glob
    files = sorted(_glob.glob(args.preds)) if any(c in args.preds for c in "*?[") else [args.preds]
    if not files:
        raise SystemExit(f"[h5->mgf] no preds files match {args.preds}")
    print(f"[h5->mgf] reading {len(files)} preds file(s)")
    n, miss = 0, 0
    with open(args.out, "w") as mgf, open(args.out + ".labels.tsv", "w") as lab:
        lab.write("spec\tsmiles\n")
        for fp in files:
            paths = []
            with h5py.File(fp, "r") as h:
                h.visititems(lambda name, obj: paths.append(name) if (isinstance(obj, h5py.Dataset) and name.endswith("/spec")) else None)
                for path in paths:
                    m = _PAT.search(path)
                    if not m:
                        continue
                    name, ce = m.group(1), m.group(2)
                    smi, prec = smi_by.get(name), prec_by.get(name)
                    if not smi or prec is None:
                        miss += 1
                        continue
                    arr = np.asarray(h[path], dtype=np.float64)
                    if arr.ndim != 2 or arr.shape[0] == 0:
                        continue
                    uid = f"{name}_ev{ce}"
                    mgf.write("BEGIN IONS\n")
                    mgf.write(f"PEPMASS={prec}\nPARENTMASS={prec}\nSMILES={smi}\nID={uid}\n"
                              f"IONIZATION=[M+H]+\nADDUCT=[M+H]+\n")
                    for mz, it in arr:
                        if it >= args.min_inten and mz > 0:
                            mgf.write(f"{mz} {it}\n")
                    mgf.write("END IONS\n")
                    lab.write(f"{uid}\t{smi}\n")
                    n += 1
    print(f"[h5->mgf] wrote {n} spectra -> {args.out} (+ .labels.tsv) from {len(files)} file(s); {miss} missing")


if __name__ == "__main__":
    main()
