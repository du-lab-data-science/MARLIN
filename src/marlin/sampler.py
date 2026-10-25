"""Mass-shell constrained block-diffusion sampler.

Molecules are written block by block; each step re-scores the block and samples its leftmost masked
position. Tokens that would push the heavy-atom mass past ``M + delta`` are pruned, ``[EOS]`` is blocked
while the molecule is too light and forced when nothing more fits, and a bracket grammar keeps the SAFE
string well formed. A candidate is accepted once it parses to a molecule within ``ppm_tol`` of ``M``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from rdkit import Chem, RDLogger

from dlm.utils.utils_chem import safe_to_smiles
from dlm.utils.benchmark_utils import (
    compute_morgan_fingerprint,
    compute_tanimoto_similarity,
    get_inchikey_first_block,
)
from mist.utils.chem_utils import mass_from_smi

from marlin import block_diffusion as bd
from marlin.chem.mass_table import min_positive_token_mass, build_valence_table, build_token_grammar
from marlin.model import MarlinDecoder

RDLogger.DisableLog("rdApp.*")

NEG_LOGIT = -1e30
BIG_LOGIT = 1e9


def load_msbd3_model(path: str, map_location=None) -> MarlinDecoder:
    map_location = map_location or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    try:
        model = MarlinDecoder.load_from_checkpoint(path, map_location=map_location, strict=True)
    except (RuntimeError, Exception):
        model = MarlinDecoder.load_from_checkpoint(path, map_location=map_location, strict=False)
    model = model.to(map_location)
    model.eval()
    if getattr(model, "ema", None) is not None:
        # evaluate with the EMA weights
        model.ema.store(model.parameters())
        model.ema.copy_to(model.parameters())
    return model


class MassShellSampler:
    def __init__(self, path: str, device=None, model: Optional[MarlinDecoder] = None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = model if model is not None else load_msbd3_model(path, self.device)
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.mask_index = self.model.mask_index
        self.bos_index = self.model.bos_index
        self.eos_index = self.model.eos_index
        self.pad_index = self.model.pad_index
        self.m_heavy = self.model.m_heavy.to(self.device)
        self.min_token_mass = min_positive_token_mass(self.m_heavy)
        # valence tables bound the hydrogen count
        val_sum, n_heavy_count = build_valence_table(self.tokenizer)
        self.val_sum = val_sum.to(self.device)
        self.n_heavy_count = n_heavy_count.to(self.device)
        self.m_H = 1.0078250319
        # bracket grammar
        paren_delta, paren_min = build_token_grammar(self.tokenizer)
        self.paren_delta = paren_delta.to(self.device)
        self.paren_min = paren_min.to(self.device)
        self.block_size = self.model.block_size
        self.max_position_embeddings = self.model.config.model.get("max_position_embeddings", 256)

        # structural special tokens are never sampled ([EOS] is handled separately)
        specials = set()
        for attr in ("pad_token_id", "bos_token_id", "mask_token_id",
                     "unk_token_id", "cls_token_id", "sep_token_id"):
            tid = getattr(self.tokenizer, attr, None)
            if tid is not None:
                specials.add(int(tid))
        specials.discard(int(self.eos_index))
        self.forbidden_token_ids = sorted(specials)

    @torch.no_grad()
    def generate(
        self,
        target_mass: float,
        isotope=None,
        fingerprint=None,
        num_candidates: int = 64,
        ppm_tol: float = 10.0,
        max_blocks: Optional[int] = None,
        softmax_temp: float = 1.0,
        enable_mass_shell_prune: bool = True,
        h_slack: float = 4.0,
        use_kv_cache: bool = True,
        fp_diversity: float = 0.0,
        fp_drop_rate: float = 0.3,
        collect_trace: bool = False,
    ) -> Dict:
        """Generate ``num_candidates`` molecules for one spectrum; returns ``{"candidates", "trace"}``."""
        device = self.device
        Bp = int(self.block_size)
        inner = Bp
        M = float(target_mass)
        delta = ppm_tol * 1e-6 * M
        if max_blocks is None:
            max_blocks = max(2, (self.max_position_embeddings - 1) // Bp)
        N = int(num_candidates)

        # same conditioning for all N candidates
        mass_t = torch.full((N,), M, device=device, dtype=torch.float32) if self.model.use_mass else None
        iso_t = None
        if self.model.use_isotope and isotope is not None:
            iso_t = torch.as_tensor(isotope, dtype=torch.float32, device=device).reshape(1, -1).expand(N, -1)
        fp_t = None
        if self.model.use_fingerprint and fingerprint is not None:
            base = torch.as_tensor(fingerprint, dtype=torch.float32, device=device).reshape(-1)
            fp_t = base.unsqueeze(0).expand(N, -1).clone()
            # conditioning diversity: the last round(fp_diversity * N) candidates each drop a random
            # fraction fp_drop_rate of the on-bits; ranking uses the full fingerprint
            n_div = int(round(float(fp_diversity) * N)) if fp_diversity and fp_drop_rate > 0 else 0
            if n_div > 0:
                on = (base > 0).nonzero().flatten()
                if on.numel() > 0:
                    div = torch.arange(N - n_div, N, device=device)
                    drop = torch.rand(n_div, on.numel(), device=device) < float(fp_drop_rate)
                    rows = div.view(-1, 1).expand(-1, on.numel())[drop]
                    cols = on.view(1, -1).expand(n_div, -1)[drop]
                    fp_t[rows, cols] = 0.0
        cond = self.model._conditioning(N, device, mass_t, iso_t, fp_t, training=False)

        kv = None
        if use_kv_cache:
            from marlin.kv_cache import BlockKVCache
            kv = BlockKVCache(self.model, N, device, cond)

        # block 0 = [BOS, MASK*(Bp-1)]
        seq = torch.full((N, Bp), self.mask_index, device=device, dtype=torch.long)
        seq[:, 0] = self.bos_index

        finished = torch.zeros(N, dtype=torch.bool, device=device)
        committed = torch.zeros(N, device=device)           # heavy-atom mass committed
        committed_val = torch.zeros(N, device=device)        # valence sum committed (for H bound)
        committed_nheavy = torch.zeros(N, device=device)     # heavy-atom count committed
        blocks_used = torch.zeros(N, dtype=torch.long, device=device)
        deadend_any = torch.zeros(N, dtype=torch.bool, device=device)
        balance = torch.zeros(N, device=device)  # running branch-parenthesis balance
        trace: List[Dict] = []

        for b in range(max_blocks):
            start, end = b * Bp, b * Bp + Bp
            finished_before = finished.clone()
            if b > 0:
                new_block = torch.full((N, Bp), self.mask_index, device=device, dtype=torch.long)
                new_block[finished] = self.pad_index
                seq = torch.cat([seq, new_block], dim=1)

            L_cur = seq.size(1)
            block_id = bd.make_block_ids(L_cur, Bp, device)
            positions = torch.arange(start, end, device=device).unsqueeze(0).expand(N, -1)
            pruned_frac_accum = 0.0
            eos_blocked_block = False
            eos_forced_block = False

            for step in range(inner):
                masked_here = (seq[:, start:end] == self.mask_index)
                active = (~finished) & masked_here.any(dim=1)
                if not bool(active.any()):
                    break

                # block logits (KV-cached or full recompute)
                if kv is not None:
                    block_valid = (seq[:, start:end] != self.pad_index)
                    blk = kv.decode(seq[:, start:end], positions, block_valid).clone()
                else:
                    pad_mask = (seq != self.pad_index)
                    add = bd.build_block_causal_mask(block_id, pad_mask, N, device)
                    blk = self.model.denoise_single(seq, add, cond=cond)[:, start:end, :].clone()

                inblock = self.m_heavy[seq[:, start:end]].sum(dim=1)
                cur = committed + inblock
                budget = (M + delta) - cur

                if self.forbidden_token_ids:
                    blk[:, :, self.forbidden_token_ids] = NEG_LOGIT

                if enable_mass_shell_prune:
                    forb = self.m_heavy.unsqueeze(0) > budget.unsqueeze(1)
                    pruned_frac_accum += float(forb.float().mean().item())
                    blk = blk.masked_fill(forb.unsqueeze(1), NEG_LOGIT)

                # EOS coupling: block [EOS] while even the most hydrogens possible,
                # sum(valence) - 2 * (n_heavy - 1) + h_slack, cannot reach M; force it when no atom fits
                inblk_val = self.val_sum[seq[:, start:end]].sum(dim=1)
                inblk_nh = self.n_heavy_count[seq[:, start:end]].sum(dim=1)
                cur_val = committed_val + inblk_val
                cur_nh = committed_nheavy + inblk_nh
                h_max = torch.clamp(cur_val - 2.0 * (cur_nh - 1.0), min=0.0)
                h_max = torch.where(cur_nh > 0, h_max, torch.zeros_like(h_max))
                max_h_mass = (h_max + float(h_slack)) * self.m_H
                too_light = (cur + max_h_mass) < (M - delta)
                force_eos = (cur + self.min_token_mass) > (M + delta)
                deadend = too_light & (budget < self.min_token_mass)
                deadend_any |= (deadend & active)
                forbid_eos = too_light & (~deadend)
                if bool(forbid_eos.any()):
                    eos_blocked_block = True
                if bool(force_eos.any()):
                    eos_forced_block = True
                eos_col = blk[:, :, self.eos_index]
                eos_col = torch.where(forbid_eos.unsqueeze(1), torch.full_like(eos_col, NEG_LOGIT), eos_col)
                eos_col = torch.where(force_eos.unsqueeze(1), torch.full_like(eos_col, BIG_LOGIT), eos_col)
                blk[:, :, self.eos_index] = eos_col

                # sample this step's position; the grammar forbids a negative bracket balance
                # and [EOS] while brackets are open
                pos = start + step
                if pos == 0:
                    continue  # BOS is fixed
                blkcol = blk[:, step, :]
                bad = (balance.unsqueeze(1) + self.paren_min.unsqueeze(0)) < -0.5
                blkcol = blkcol.masked_fill(bad, NEG_LOGIT)
                unbal = balance.abs() > 0.5
                blkcol[:, self.eos_index] = torch.where(
                    unbal, torch.full_like(blkcol[:, self.eos_index], NEG_LOGIT), blkcol[:, self.eos_index])
                probs = torch.softmax(blkcol.float() / max(float(softmax_temp), 1e-6), dim=-1)
                tok = torch.multinomial(probs, num_samples=1).squeeze(1)
                upd = active & (seq[:, pos] == self.mask_index)
                seq[upd, pos] = tok[upd]
                balance = balance + torch.where(upd, self.paren_delta[tok], torch.zeros_like(balance))
                placed_eos = (seq[:, pos] == self.eos_index) & (~finished)
                if bool(placed_eos.any()):
                    bs = seq[:, start:end].clone()
                    rem = bs == self.mask_index
                    bs[placed_eos.unsqueeze(1) & rem] = self.pad_index
                    seq[:, start:end] = bs
                    finished = finished | placed_eos

            # pad positions left masked
            leftover = seq[:, start:end] == self.mask_index
            if bool(leftover.any()):
                bs = seq[:, start:end].clone()
                bs[leftover] = self.pad_index
                seq[:, start:end] = bs

            # cache the finished block's keys and values
            if kv is not None:
                kv.commit(seq[:, start:end], positions, (seq[:, start:end] != self.pad_index))

            # accept a candidate as soon as it parses to a molecule within ppm_tol of M
            if not bool(finished.all()):
                # only parse rows that could already reach M once hydrogens are added
                cand = (~finished)
                cur_heavy_b = committed + self.m_heavy[seq[:, start:end]].sum(dim=1)
                cv = committed_val + self.val_sum[seq[:, start:end]].sum(dim=1)
                cnh = committed_nheavy + self.n_heavy_count[seq[:, start:end]].sum(dim=1)
                hmx = torch.clamp(cv - 2.0 * (cnh - 1.0), min=0.0)
                hmx = torch.where(cnh > 0, hmx, torch.zeros_like(hmx))
                mhm = (hmx + float(h_slack)) * self.m_H
                cand = cand & ((cur_heavy_b + mhm) >= (M - delta))
                idxs = [i for i in range(N) if bool(cand[i])]
                if idxs:
                    decoded = self.tokenizer.batch_decode(seq[idxs], skip_special_tokens=True)
                    for k, i in enumerate(idxs):
                        smi = self._to_smiles(decoded[k])
                        if not smi:
                            continue
                        mp = float(mass_from_smi(smi))
                        if mp > 0 and abs(mp - M) / M * 1e6 <= ppm_tol:
                            finished[i] = True

            committed = committed + self.m_heavy[seq[:, start:end]].sum(dim=1)
            committed_val = committed_val + self.val_sum[seq[:, start:end]].sum(dim=1)
            committed_nheavy = committed_nheavy + self.n_heavy_count[seq[:, start:end]].sum(dim=1)
            blocks_used = blocks_used + (~finished_before).long()

            if collect_trace:
                trace.append({
                    "block": b,
                    "committed_heavy_mean": float(committed.mean().item()),
                    "pruned_token_frac": pruned_frac_accum / max(inner, 1),
                    "eos_blocked": eos_blocked_block,
                    "eos_forced": eos_forced_block,
                    "n_active": int((~finished_before).sum().item()),
                })

            if bool(finished.all()):
                break

        candidates = self._decode_and_accept(seq, M, ppm_tol, blocks_used, deadend_any)
        return {"candidates": candidates, "trace": trace}

    def _to_smiles(self, safe_str):
        """Decode a SAFE string to the (largest-fragment) SMILES, or None."""
        if not safe_str:
            return None
        smiles = safe_to_smiles(safe_str, fix=True)
        if smiles:
            smiles = sorted(smiles.split("."), key=len)[-1]
        return smiles or None

    def _decode_and_accept(self, seq, M, ppm_tol, blocks_used, deadend_any) -> List[Dict]:
        decoded = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
        out = []
        for i, safe_str in enumerate(decoded):
            smiles = self._to_smiles(safe_str)
            m_pred = float(mass_from_smi(smiles)) if smiles else 0.0
            err_ppm = abs(m_pred - M) / M * 1e6 if (smiles and M > 0) else float("inf")
            accepted = bool(smiles) and (err_ppm <= ppm_tol)
            out.append({
                "safe": safe_str,
                "smiles": smiles,
                "mass_pred": m_pred,
                "mass_err_ppm": err_ppm,
                "accepted": accepted,
                "n_blocks": int(blocks_used[i].item()),
                "deadend": bool(deadend_any[i].item()),
            })
        return out

    def rank(
        self,
        candidates: List[Dict],
        fingerprint_target,
        fp_bits: int = 4096,
        fp_radius: int = 2,
        require_accept: bool = True,
    ) -> List[Dict]:
        """Rank accepted candidates by Tanimoto to the predicted fingerprint, one per InChIKey connectivity."""
        pool = [c for c in candidates if (c.get("accepted") or not require_accept)]
        pool = [c for c in pool if c.get("smiles")]
        seen = set()
        ranked = []
        for c in pool:
            cand_fp = compute_morgan_fingerprint(c["smiles"], fp_bits, fp_radius)
            tani = compute_tanimoto_similarity(fingerprint_target, cand_fp) if cand_fp is not None else 0.0
            c = dict(c)
            c["tanimoto_to_f"] = float(tani)
            c["score"] = float(tani)
            ranked.append(c)
        ranked.sort(key=lambda x: x["score"], reverse=True)
        deduped = []
        for c in ranked:
            try:
                mol = Chem.MolFromSmiles(c["smiles"])
                key = get_inchikey_first_block(Chem.MolToInchiKey(mol)) if mol else c["smiles"]
            except Exception:
                key = c["smiles"]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        for rank_i, c in enumerate(deduped):
            c["rank"] = rank_i + 1
        return deduped
