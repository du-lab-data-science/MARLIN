"""Turn FRIGID's ``canopus_aug_filt`` labels into an ms-pred labels.tsv for re-simulating raw ICEBERG spectra.

FRIGID's set only has subformula-annotated peaks, which carry the formula, so the raw spectra are re-simulated.

  python -m marlin.prep_iceberg_aug --aug-labels <canopus_aug_filt>/labels.csv \
      --out-dir ms-pred/data/spec_datasets/canopus_aug_raw

Then run ms-pred's ``predict_smis.py --sparse-out`` and ``marlin.iceberg_h5_to_mgf``.
"""

import argparse
import csv
import os
import re
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from marlin.dreams_precompute import _formula_monoiso, _PROTON

_EV = re.compile(r"_ev-?(\d+)")

# adduct -> (m/z added to the neutral mass, spec-id tag)
_ADDUCT = {
    "[M+H]+":  (_PROTON, "MpHp"),
    "[M+Na]+": (22.989221, "MpNap"),
}


def main():
    ap = argparse.ArgumentParser(description="canopus_aug_filt labels.csv -> ms-pred ICEBERG labels.tsv")
    ap.add_argument("--aug-labels", required=True, help="FRIGID canopus_aug_filt labels.csv")
    ap.add_argument("--out-dir", required=True, help="ms-pred dataset dir (writes labels.tsv + splits/split.tsv)")
    ap.add_argument("--dataset-name", default="canopus_aug_raw")
    ap.add_argument("--limit-molecules", type=int, default=None, help="cap distinct (mol,instrument) groups")
    args = ap.parse_args()

    # group rows by molecule, instrument and adduct
    groups = {}
    n = 0
    with open(args.aug_labels, encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            n += 1
            ik = str(row.get("inchikey", "")).split("-")[0]
            instr = row.get("instrument", "")
            ion = row.get("ionization", "[M+H]+")
            smi = row.get("smiles"); form = row.get("formula")
            if not ik or not smi or not form or ion not in _ADDUCT:
                continue
            m = _EV.search(row.get("spec", ""))
            ce = m.group(1) if m else None
            key = (ik, instr, ion)
            g = groups.get(key)
            if g is None:
                if args.limit_molecules and len(groups) >= args.limit_molecules:
                    continue
                g = groups[key] = {"smiles": smi, "formula": form, "inchikey": row.get("inchikey", ""),
                                   "instrument": instr, "ionization": ion, "ce": set()}
            if ce is not None:
                g["ce"].add(ce)
    print(f"[prep] read {n} rows -> {len(groups)} (molecule, instrument) groups")

    os.makedirs(os.path.join(args.out_dir, "splits"), exist_ok=True)
    lbl_path = os.path.join(args.out_dir, "labels.tsv")
    split_path = os.path.join(args.out_dir, "splits", "split.tsv")
    cols = ["dataset", "spec", "ionization", "formula", "smiles", "inchikey",
            "instrument", "collision_energies", "precursor"]
    n_out = 0
    with open(lbl_path, "w", newline="") as lf, open(split_path, "w", newline="") as sf:
        lw = csv.writer(lf, delimiter="\t"); lw.writerow(cols)
        sw = csv.writer(sf, delimiter="\t"); sw.writerow(["spec", "split"])
        for (ik, instr, ion), g in groups.items():
            delta, ion_tag = _ADDUCT[ion]
            try:
                prec = round(_formula_monoiso(g["formula"]) + delta, 5)
            except Exception:
                continue
            ce_list = sorted(g["ce"], key=lambda x: int(x)) or ["nan"]
            spec_id = f"{ik}_{instr}_{ion_tag}"
            lw.writerow([args.dataset_name, spec_id, ion, g["formula"], g["smiles"],
                         g["inchikey"], instr, str(ce_list), prec])
            sw.writerow([spec_id, "test"])   # predict_smis predicts the "test" subset
            n_out += 1
    print(f"[prep] wrote {n_out} rows -> {lbl_path}  (+ split {split_path})")
    print(f"[prep] NEXT: predict_smis --gen-checkpoint <iceberg model1> --inten-checkpoint <model2> "
          f"--sparse-out --dataset-labels {lbl_path} --subset-datasets test_only -> preds.hdf5 -> iceberg_h5_to_mgf")


if __name__ == "__main__":
    main()
