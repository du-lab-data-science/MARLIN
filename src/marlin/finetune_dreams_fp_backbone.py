"""Fine-tune the DreaMS backbone with a fingerprint head (DreaMS environment).

Train in float32 (no ``--amp``): the embedding cache from ``dreams_precompute`` is float32, and a head
trained under bf16 does not transfer to it. The released encoder was trained with
``scripts/train_dreams_encoder.sh``; then embed the spectra with ``marlin.dreams_precompute``.
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
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

from marlin.dreams_adapter import DreamsEmbedder, DEFAULT_EMB_CKPT, DEFAULT_SSL_CKPT
from marlin.dreams_precompute import parse_ms_file, iter_mgf_blocks, iter_subform_blocks, _load_aug_smiles
from marlin.dreams_fphead import DreamsFPHead


def morgan(smiles, bits, radius):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    a = np.zeros((bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius, bits), a)
    return a


def tanimoto_rows(pred_bin, true_bin):
    inter = (pred_bin * true_bin).sum(1)
    union = ((pred_bin + true_bin) > 0).sum(1)
    return float((inter / np.maximum(union, 1)).mean())


def best_threshold(probs, true_bin, grid=None):
    grid = grid if grid is not None else np.arange(0.02, 0.62, 0.02)
    bt, bv = 0.5, -1.0
    for t in grid:
        v = tanimoto_rows((probs >= t).astype(np.float32), true_bin)
        if v > bv:
            bv, bt = v, float(t)
    return bt, bv


def set_backbone_trainable(backbone, unfreeze_last_k, freeze_all):
    for p in backbone.parameters():
        p.requires_grad = False
    unused = ("ff_out", "ro_out", "mz_masking_loss")
    if freeze_all:
        return
    if unfreeze_last_k is None or unfreeze_last_k < 0:
        for n, p in backbone.named_parameters():
            p.requires_grad = not n.startswith(unused)
        return
    keep = set(range(max(0, backbone.n_layers - unfreeze_last_k), backbone.n_layers))
    for n, p in backbone.named_parameters():
        if n.startswith(unused):
            continue
        if any(f"ffs.{i}." in n or f"atts.{i}." in n or f"scales.{i}." in n for i in keep):
            p.requires_grad = True


def load_split(labels_file, split_file):
    labels = pd.read_csv(labels_file, sep="\t")
    splits = pd.read_csv(split_file, sep="\t")
    sc = "split" if "split" in splits.columns else splits.columns[-1]
    nc = "name" if "name" in splits.columns else splits.columns[0]
    s2split = dict(zip(splits[nc].astype(str), splits[sc].astype(str)))
    spec2smi = dict(zip(labels["spec"].astype(str), labels["smiles"].astype(str)))
    return spec2smi, s2split


def build_dataset(spec_folder, spec2smi, s2split, which, embedder, bits, radius, limit=None):
    X, Y = [], []
    for fp in tqdm(sorted(glob.glob(os.path.join(spec_folder, "*.ms"))), desc=f"prep[{which}]"):
        sid = os.path.splitext(os.path.basename(fp))[0]
        if s2split.get(sid) != which or sid not in spec2smi:
            continue
        y = morgan(spec2smi[sid], bits, radius)
        if y is None:
            continue
        prec_mz, peaks = parse_ms_file(fp)
        if prec_mz is None:
            continue
        X.append(embedder.preprocess(peaks, prec_mz)); Y.append(y)
        if limit and len(X) >= limit:
            break
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.stack(Y).astype(np.float32)


def build_synth_dataset(mgf, aug_labels, embedder, bits, radius, limit=None):
    """Simulated MGF spectra -> preprocessed peaks and bit-packed fingerprint targets."""
    spec2smi = _load_aug_smiles(aug_labels)
    X, Yp = [], []
    for sid, prec_mz, peaks in tqdm(iter_mgf_blocks(mgf), desc="prep[synth]"):
        smi = spec2smi.get(sid)
        if not smi:
            continue
        y = morgan(smi, bits, radius)
        if y is None:
            continue
        X.append(embedder.preprocess(peaks, prec_mz))
        Yp.append(np.packbits(y.astype(np.uint8)))
        if limit and len(X) >= limit:
            break
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.stack(Yp)


def build_synth_dataset_from_subforms(subform_dir, labels_csv, embedder, bits, radius, limit=None):
    """Like ``build_synth_dataset``, reading MIST subformula JSON files instead of an MGF."""
    spec2smi = _load_aug_smiles(labels_csv)
    X, Yp = [], []
    for sid, prec_mz, peaks in tqdm(iter_subform_blocks(subform_dir), desc="prep[synth-subform]"):
        smi = spec2smi.get(sid)
        if not smi:
            continue
        y = morgan(smi, bits, radius)
        if y is None:
            continue
        X.append(embedder.preprocess(peaks, prec_mz))
        Yp.append(np.packbits(y.astype(np.uint8)))
        if limit and len(X) >= limit:
            break
    if not X:
        return None, None
    return np.stack(X).astype(np.float32), np.stack(Yp)


def main():
    ap = argparse.ArgumentParser(description="Fine-tune DreaMS backbone + head for fingerprints (BCE)")
    ap.add_argument("--spec-folder", default="data/canopus/spec_files")
    ap.add_argument("--labels", default="data/canopus/labels.tsv")
    ap.add_argument("--split-file", default="data/canopus/splits/canopus_hplus_100_0.tsv")
    ap.add_argument("--out-dir", default="train_results/dreams_fp_backbone")
    ap.add_argument("--emb-ckpt", default=DEFAULT_EMB_CKPT)
    ap.add_argument("--ssl-ckpt", default=DEFAULT_SSL_CKPT)
    ap.add_argument("--n-highest-peaks", type=int, default=100)
    ap.add_argument("--bits", type=int, default=4096)
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--emb-kind", default="backbone", choices=["backbone", "all_peaks"],
                    help="backbone = precursor token; all_peaks = attention-pool over ALL DreaMS peak embeddings")
    # ICEBERG-simulated spectra mixed into training
    ap.add_argument("--synthetic-mgf", default=None,
                    help="ICEBERG full_out.mgf; streamed + DreaMS-preprocessed, mixed into TRAIN at --frac-orig")
    ap.add_argument("--synthetic-subform-dir", default=None,
                    help="DO NOT USE for the formula-free encoder. Subform output_tbl.mz are THEORETICAL "
                         "formula masses (== mono_mass) and the peak SET is filtered to sub-formulae of the "
                         "precursor molecular formula -> this LEAKS the formula into the spectrum. Only the raw "
                         "simulated ICEBERG MGF (--synthetic-mgf) is formula-free aug. (Flag kept for non-"
                         "formula-free ablations only.)")
    ap.add_argument("--subform-labels", default=None,
                    help="labels (TSV) for --synthetic-subform-dir (spec->smiles)")
    ap.add_argument("--aug-labels", default=None,
                    help="labels table of the synthetic MGF (spec -> smiles); required with --synthetic-mgf")
    ap.add_argument("--synthetic-limit", type=int, default=None, help="cap synthetic pool size (RAM/startup)")
    ap.add_argument("--frac-orig", type=float, default=1.0,
                    help="fraction of REAL examples per epoch; <1 mixes synthetic (resampled each epoch); 1.0=off")
    # head
    ap.add_argument("--iterative", default="none", choices=["none", "growing"])
    ap.add_argument("--num-splits", type=int, default=4)
    ap.add_argument("--reduce-factor", type=int, default=2)
    ap.add_argument("--iterative-loss-weight", type=float, default=0.4)
    ap.add_argument("--loss-fn", default="bce", choices=["bce", "cosine"])
    # head size and regularization
    ap.add_argument("--hidden-dim", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    # backbone freezing
    ap.add_argument("--unfreeze-last-k", type=int, default=-1, help="-1=full backbone; K=top-K blocks")
    ap.add_argument("--freeze-backbone", action="store_true", help="train the head only (frozen backbone)")
    # optim
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--pos-weight", type=float, default=0.0)
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast. WARNING: the deployment cache (dreams_precompute) is float32; "
                         "DreaMS's large m/z features quantize coarsely in bf16, so a head trained with "
                         "--amp COLLAPSES on the float32 cache (bf16 val 0.43 -> float32 deploy 0.18). "
                         "Leave OFF so train precision == deploy precision (fp32 is fine; only fp16 overflows).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.synthetic_mgf and args.frac_orig < 1.0 and not args.aug_labels:
        ap.error("--synthetic-mgf needs --aug-labels (the MGF's spec -> smiles table)")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder = DreamsEmbedder(emb_ckpt=args.emb_ckpt, ssl_ckpt=args.ssl_ckpt,
                              emb_kind="backbone", n_highest_peaks=args.n_highest_peaks, device=dev)
    backbone = embedder.backbone
    set_backbone_trainable(backbone, args.unfreeze_last_k, args.freeze_backbone)
    all_peaks = args.emb_kind == "all_peaks"
    growing = args.iterative == "growing"
    head = DreamsFPHead(in_dim=embedder.d_model, bits=args.bits, hidden_dim=args.hidden_dim,
                        depth=args.depth, dropout=args.dropout, set_input=all_peaks,
                        iterative=args.iterative, num_splits=args.num_splits,
                        reduce_factor=args.reduce_factor).to(dev)
    n_bb = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    mode = "head-only" if args.freeze_backbone else ("full" if args.unfreeze_last_k < 0 else f"last-{args.unfreeze_last_k}")
    print(f"[fpft] d_model={embedder.d_model} head_params={sum(p.numel() for p in head.parameters())/1e6:.2f}M "
          f"trainable_backbone={n_bb/1e6:.1f}M mode={mode}", flush=True)

    spec2smi, s2split = load_split(args.labels, args.split_file)
    Xtr, Ytr = build_dataset(args.spec_folder, spec2smi, s2split, "train", embedder, args.bits, args.radius, args.limit)
    Xva, Yva = build_dataset(args.spec_folder, spec2smi, s2split, "val", embedder, args.bits, args.radius, args.limit)
    if Xtr is None:
        raise SystemExit("No training pairs.")
    print(f"[fpft] train={Xtr.shape[0]} val={0 if Xva is None else Xva.shape[0]}", flush=True)
    Xtr_t = torch.from_numpy(Xtr); Ytr_t = torch.from_numpy(Ytr)
    if Xva is not None:
        Xva_t = torch.from_numpy(Xva)

    # simulated spectra are resampled into training each epoch; validation stays real
    Xsyn = Ysyn_p = None
    if args.frac_orig < 1.0 and (args.synthetic_subform_dir or args.synthetic_mgf):
        if args.synthetic_subform_dir:
            Xsyn, Ysyn_p = build_synth_dataset_from_subforms(
                args.synthetic_subform_dir, args.subform_labels, embedder,
                args.bits, args.radius, args.synthetic_limit)
        else:
            Xsyn, Ysyn_p = build_synth_dataset(args.synthetic_mgf, args.aug_labels, embedder,
                                               args.bits, args.radius, args.synthetic_limit)
        assert Xsyn is not None and Xsyn.shape[1:] == Xtr.shape[1:], "synth/real preprocessed shape mismatch"
        print(f"[fpft] synthetic pool={Xsyn.shape[0]} (frac_orig={args.frac_orig}, "
              f"~{int(round(Xtr.shape[0]*(1-args.frac_orig)/max(args.frac_orig,1e-6)))} mixed/epoch)", flush=True)

    groups = [{"params": list(head.parameters()), "lr": args.lr_head}]
    bb_params = [p for p in backbone.parameters() if p.requires_grad]
    if bb_params:
        groups.append({"params": bb_params, "lr": args.lr_backbone})
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    pw = torch.full((args.bits,), args.pos_weight, device=dev) if args.pos_weight > 0 else None
    _bce_logit = nn.BCEWithLogitsLoss(pos_weight=pw)
    _cos = nn.CosineSimilarity(dim=-1)
    use_amp = args.amp and dev.type == "cuda"

    def head_in(spec):                                     # backbone features for the head
        full = backbone(spec, None)
        if all_peaks:
            return full, (spec[:, :, 0] != 0).float()      # peak set + mask
        return full[:, 0, :], None                         # precursor token

    def prob_loss(p, t):                                   # growing outputs / intermediates (probs)
        if args.loss_fn == "cosine":
            return (1.0 - _cos(p, t.float())).mean()
        return torch.nn.functional.binary_cross_entropy(p.clamp(1e-6, 1 - 1e-6), t.float())

    def logit_loss(z, t):                                  # plain head (logits)
        if args.loss_fn == "cosine":
            return (1.0 - _cos(torch.sigmoid(z), t.float())).mean()
        return _bce_logit(z, t)

    def step_loss(spec, yb):
        feats, mask = head_in(spec)
        if growing:
            outs = head.forward_all(feats, mask)
            loss = prob_loss(outs[-1], yb)
            aux, cur = 0.0, yb                             # MIST multi-resolution iterative loss
            for ip in reversed(outs[:-1]):
                ts = ip.shape[-1]
                bi, ji = torch.where(cur > 0.5)
                nt = torch.zeros_like(ip); nt[bi, ji % ts] = 1.0
                aux = aux + prob_loss(ip, nt); cur = nt
            return loss + args.iterative_loss_weight * aux
        return logit_loss(head(feats, mask), yb)

    best = {"tani": -1.0, "thr": 0.5, "epoch": -1, "head": None, "bb": None}
    bad = 0
    for ep in range(1, args.epochs + 1):
        # all real spectra plus a fresh simulated sample
        if Xsyn is not None:
            k = min(int(round(Xtr_t.shape[0] * (1 - args.frac_orig) / max(args.frac_orig, 1e-6))), Xsyn.shape[0])
            si = np.random.choice(Xsyn.shape[0], k, replace=False)
            Ys = np.unpackbits(Ysyn_p[si], axis=1)[:, :args.bits].astype(np.float32)
            Xe = torch.cat([Xtr_t, torch.from_numpy(Xsyn[si])], 0)
            Ye = torch.cat([Ytr_t, torch.from_numpy(Ys)], 0)
        else:
            Xe, Ye = Xtr_t, Ytr_t
        n = Xe.shape[0]
        head.train(); backbone.train() if bb_params else backbone.eval()
        perm = torch.randperm(n); tot = 0.0; opt.zero_grad()
        for bi, i in enumerate(range(0, n, args.batch_size)):
            idx = perm[i:i + args.batch_size]
            xb = Xe[idx].to(dev); yb = Ye[idx].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = step_loss(xb, yb) / args.grad_accum
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            loss.backward()
            if (bi + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], args.max_grad_norm)
                opt.step(); opt.zero_grad()
            tot += loss.item() * args.grad_accum * idx.numel()

        if Xva is None:
            continue
        head.eval(); backbone.eval()
        probs = []
        with torch.no_grad():
            for i in range(0, Xva_t.shape[0], args.batch_size):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    feats, mask = head_in(Xva_t[i:i + args.batch_size].to(dev))
                    p = head.predict_proba(feats, mask)
                probs.append(p.float().cpu().numpy())
        probs = np.concatenate(probs, 0)
        thr, tani = best_threshold(probs, Yva)
        print(f"[fpft] ep{ep:3d} loss={tot/n:.4f} val_tani={tani:.4f}@thr{thr:.2f} (best {best['tani']:.4f})", flush=True)
        if tani > best["tani"]:
            best.update(tani=tani, thr=thr, epoch=ep,
                        head={k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                        bb=({k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()} if bb_params else None))
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[fpft] early stop ep{ep} (best val_tani={best['tani']:.4f}@ep{best['epoch']})")
                break

    os.makedirs(args.out_dir, exist_ok=True)
    if best["head"] is not None:
        head.load_state_dict(best["head"])
    head.save(os.path.join(args.out_dir, "head.pt"), emb_kind=args.emb_kind, threshold=best["thr"],
              extra={"val_tanimoto": best["tani"], "val_epoch": best["epoch"], "backbone_finetuned": best["bb"] is not None})
    print(f"[fpft] saved head -> {os.path.join(args.out_dir, 'head.pt')} (val_tani={best['tani']:.4f} thr={best['thr']:.2f})")
    if best["bb"] is not None:
        backbone.load_state_dict(best["bb"])
        torch.save({"state_dict": backbone.state_dict(),
                    "meta": {"emb_kind": args.emb_kind, "n_highest_peaks": args.n_highest_peaks}},
                   os.path.join(args.out_dir, "backbone_finetuned_fp.pt"))
        print(f"[fpft] saved fine-tuned backbone -> {os.path.join(args.out_dir, 'backbone_finetuned_fp.pt')}")
    print("[fpft] NEXT: dreams_precompute --finetuned-backbone <backbone_finetuned_fp.pt> -> cache; "
          "then infer --encoder dreams with that cache + this head.pt")


if __name__ == "__main__":
    main()
