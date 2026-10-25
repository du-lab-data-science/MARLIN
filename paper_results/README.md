# Paper predictions

Per-spectrum predictions of the released MARLIN configurations on the NPLIB1 test split (803 spectra). Each folder
holds `predictions.jsonl.gz` (one record per spectrum: the true structure and the ranked accepted candidates) and,
for the two encoder configurations, `mces_cache_k10.jsonl.gz` (Top-1 / Top-10 MCES per spectrum).

Recompute the metrics without a GPU:

```bash
python -m marlin.summarize paper_results/marlin_dreams paper_results/marlin_mist
python -m marlin.summarize paper_results/abl_*
```

| Folder | Configuration | Command (see `scripts/reproduce_paper.sh`) |
|---|---|---|
| `marlin_dreams` | MARLIN with the DreaMS encoder (also the full model for the ablations) | `infer --config infer_nplib1_dreams.yaml` |
| `marlin_mist` | MARLIN with the MIST encoder | `infer --config infer_nplib1_mist.yaml` |
| `abl_dreams_wo_conditioning_diversity` | ablation: w/o conditioning diversity | `--fp-diversity 0.0` |
| `abl_dreams_wo_mass_conditioning` | ablation: w/o mass conditioning and heavy-atom prune | `--no-mass-conditioning --no-mass-shell-prune` |
| `abl_dreams_wo_prune` | ablation: w/o heavy-atom prune | `--no-mass-shell-prune` |

Accuracy is computed over all 803 spectra; Tanimoto and MCES over the spectra with at least one accepted
candidate (MARLIN returns no candidate when none of its decodes matches the precursor mass within 10 ppm).
MCES was computed with `marlin.compute_mces` using the HiGHS solver. All runs used the fixed seed 42.
