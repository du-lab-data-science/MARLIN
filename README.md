# MARLIN

**De novo molecular structure elucidation from tandem mass spectra without a ground-truth formula**

Xujun Che, Xiuxia Du, Depeng Xu (University of North Carolina at Charlotte). To appear at IEEE BIBM 2026.
Preprint: [arXiv:2607.04774](https://arxiv.org/abs/2607.04774).

Most de novo methods assume the molecular formula of the unknown is given. MARLIN removes that assumption. A spectral
encoder predicts a Morgan fingerprint from the MS/MS spectrum, and a block-diffusion language model writes candidate
structures as SAFE strings, conditioned only on that fingerprint and the precursor mass. A mass-shell constraint keeps
every partial molecule within the precursor-mass budget, and a candidate is accepted only if its exact monoisotopic mass
matches the precursor within 10 ppm. With the DreaMS encoder, no molecular formula is predicted or used anywhere in the
pipeline.

This repository contains the code, configurations, trained weights and saved predictions behind the paper's results on
NPLIB1.

## Contents

```
src/marlin/          MARLIN: decoder model, block diffusion, mass-shell sampler, encoders, training and evaluation
src/marlin/configs/  infer_nplib1_dreams.yaml, infer_nplib1_mist.yaml, train_marlin_decoder.yaml
src/dlm/, src/mist/  the parts of FRIGID's backbone and MIST encoder code that MARLIN builds on (see NOTICE)
scripts/             download, reproduce, MCES and training entry points
requirements*.txt    Python dependencies (requirements-lock.txt pins the exact versions)
data/marlin/         MIST-CF predicted formulas, decoder exclusion list, DreaMS embeddings, MIST-CF subformulae
weights/             checkpoints (downloaded by scripts/download_weights.sh; SHA256SUMS lists all of them)
paper_results/       per-spectrum predictions of the released configurations on the NPLIB1 test split
```

## Installation

```bash
conda create -n marlin python=3.10 pip
conda activate marlin
pip install -r requirements-lock.txt         # the exact package versions behind the paper (PyTorch 2.6, CUDA 12.4)
pip install -e . --no-deps                   # makes `python -m marlin.<module>` importable
```

`requirements.txt` lists the top-level packages. All commands below run from the repository root.

A second environment with DreaMS is only needed to fine-tune the DreaMS encoder or to embed new spectra; reproducing
the paper on NPLIB1 uses the released embedding cache instead.

```bash
conda create -n marlin-dreams python=3.11 pip
conda activate marlin-dreams
pip install "dreams @ git+https://github.com/pluskal-lab/DreaMS.git@f3b87019835719f840acebda24789cd656b8fd8b"
pip install -e . --no-deps
```

## Data and weights

```bash
bash scripts/download_data.sh       # NPLIB1 (MIST's CANOPUS export) -> data/canopus, checksum-verified
bash scripts/download_weights.sh    # MARLIN checkpoints, MIST encoder, DreaMS embeddings, SAFE tokenizer
#   --dreams  also the official DreaMS weights (only to embed new spectra)
```

| File | What it is | Source |
|---|---|---|
| `weights/marlin_decoder.ckpt` | MARLIN block-diffusion decoder (step 1,930,000; EMA weights used at inference) | [MARLIN release](https://doi.org/10.5281/zenodo.23632849) |
| `weights/dreams_fp/backbone_finetuned_fp.pt`, `head.pt` | DreaMS backbone fine-tuned for fingerprints, and its fingerprint head | [MARLIN release](https://doi.org/10.5281/zenodo.23632849) |
| `data/marlin/dreams_embeddings_nplib1.npz` | that backbone's embedding of every NPLIB1 spectrum | [MARLIN release](https://doi.org/10.5281/zenodo.23632849) |
| `weights/mistcf/mistcf_nplib1.ckpt` | MIST-CF retrained on the `canopus_hplus_100_0` training split | [MARLIN release](https://doi.org/10.5281/zenodo.23632849) |
| `data/marlin/top5_formulas_natural.json` | its top-5 formula predictions for the split (top-1 correct on 79.0% of test spectra) | this repository |
| `data/marlin/subformulae_mistcf_top1/` | peak subformulae for the MIST encoder, from the top-1 predicted formula | [MARLIN release](https://doi.org/10.5281/zenodo.23632849) |
| `data/marlin/exclude_inchikeys_valtest.csv` | NPLIB1 + MassSpecGym val/test InChIKeys removed from decoder training | this repository |
| `weights/frigid/mist_canopus.pt` | MIST spectrum encoder used by MARLIN (MIST) | [FRIGID release](https://zenodo.org/records/19685145) |
| `weights/frigid/DLM.ckpt` | decoder checkpoint that MARLIN's decoder training is warm-started from | [FRIGID release](https://zenodo.org/records/19685145) |
| `weights/dreams/ssl_model.ckpt`, `embedding_model.ckpt` | official DreaMS weights (only to embed new spectra) | [DreaMS](https://zenodo.org/records/10997887) |

## Reproducing the paper

Inference runs on one GPU per process. On an NVIDIA L40S with 384 decodes per spectrum, MARLIN takes about 34 s per
spectrum (the 803 test spectra take about 8 GPU-hours). Every command accepts
`--num-shards N --shard-idx i` to split the test set across GPUs; merge the shards with
`python -m marlin.merge_shards results/<name>`.

```bash
bash scripts/reproduce_paper.sh main          # MARLIN with the DreaMS and the MIST encoder
bash scripts/reproduce_paper.sh ablations     # the inference-time ablations
bash scripts/compute_mces.sh results/marlin_dreams      # MCES (CPU; HiGHS, or SOLVER=GUROBI)
python -m marlin.summarize results/marlin_dreams results/marlin_mist
```

Both encoder configurations are single `marlin.infer` calls. The DreaMS encoder reads raw peaks only, so that
configuration is fully formula-free; the MIST encoder featurizes each peak with subformulae of a MIST-CF predicted formula:

```bash
python -m marlin.infer --config src/marlin/configs/infer_nplib1_dreams.yaml --output-dir results/marlin_dreams
python -m marlin.infer --config src/marlin/configs/infer_nplib1_mist.yaml   --output-dir results/marlin_mist
```

Expected output of `marlin.summarize` (NPLIB1 test, 803 spectra; accuracy over all spectra, MCES and Tanimoto over
the spectra that produced a candidate):

| Method | Top-1 acc. | Top-1 MCES | Top-1 Tanimoto | Top-10 acc. | Top-10 MCES | Top-10 Tanimoto |
|---|---|---|---|---|---|---|
| MARLIN (DreaMS) | 16.94% | 6.79 | 0.55 | 23.54% | 5.83 | 0.60 |
| MARLIN (MIST) | 19.18% | 8.59 | 0.51 | 26.65% | 7.40 | 0.57 |

Decoding is stochastic. The runs use a fixed seed, but a different GPU or library version changes the random stream,
so expect small run-to-run differences. To check the paper's numbers without a GPU, run `marlin.summarize` on the
saved predictions in `paper_results/`.

Ablation flags (they apply to either config):

| Configuration | Flags |
|---|---|
| w/o conditioning diversity | `--fp-diversity 0.0` |
| w/o mass conditioning and heavy-atom prune | `--no-mass-conditioning --no-mass-shell-prune` |
| w/o heavy-atom prune | `--no-mass-shell-prune` |

## Evaluation protocol notes

* **Test set.** The `canopus_hplus_100_0` split has 819 test spectra; the MIST data loader used by both MARLIN
  configurations yields 803 of them.
* **Precursor mass.** The released configs set `mass_source: true_mass`, the neutral monoisotopic mass of the reference
  structure, which is what the paper runs used. For 788 of the 803 test spectra this equals the mass derived from the
  recorded NPLIB1 precursor m/z, because NPLIB1 stores the theoretical [M+H]+ value. For the other 15, the recorded value
  differs from the theoretical [M+H]+ by exactly one proton mass. For new data use `--mass-source precursor` (the adduct
  comes from the spectrum metadata, default `[M+H]+`).
* **No test leakage.** The decoder never trains on a molecule whose 2D InChIKey is in the NPLIB1 or MassSpecGym val/test
  splits (`data/marlin/exclude_inchikeys_valtest.csv`). MIST-CF was retrained on the training split only. The ICEBERG-
  simulated molecules used to fine-tune DreaMS share no molecule with the NPLIB1 validation or test sets.

## Running on your own spectra

`marlin.infer` is an evaluation script: it scores predictions against reference structures. For unlabeled spectra,
use `marlin.predict` with the formula-free DreaMS encoder. Spectra can be a folder of SIRIUS `.ms` files or one MGF
file; each needs a precursor m/z and, ideally, its adduct (default `[M+H]+`).

```bash
# 1) DreaMS environment: embed the spectra with the fine-tuned DreaMS backbone
#    (needs the official DreaMS weights: bash scripts/download_weights.sh --dreams)
python -m marlin.dreams_precompute --source mgf --mgf my_spectra.mgf --emb-kind backbone \
    --finetuned-backbone weights/dreams_fp/backbone_finetuned_fp.pt --out my_spectra_emb.npz
# 2) main environment: generate and rank candidates
python -m marlin.predict --spectra my_spectra.mgf --embeddings my_spectra_emb.npz --out results/my_spectra
```

`results/my_spectra/predictions.tsv` lists the top 10 candidates per spectrum (SMILES, formula, ppm mass error,
Tanimoto to the predicted fingerprint). The neutral mass is computed from the precursor m/z and adduct; decoding
settings come from `src/marlin/configs/infer_nplib1_dreams.yaml`. The MIST encoder additionally needs peak
subformulae of a predicted formula, which requires SIRIUS and [MIST-CF](https://github.com/samgoldman97/mist-cf);
the DreaMS route avoids that.

## Training

**Decoder.** Self-supervised on the SAFE-GPT corpus (no spectra): the mass, isotope features and fingerprint
conditions are computed from each structure with RDKit, a fraction 0.5 of fingerprints are corrupted by dropping and
adding a U[0.1, 0.3] fraction of bits, and training is warm-started from FRIGID's decoder.

```bash
huggingface-cli download datamol-io/safe-gpt --repo-type dataset \
    --revision 16d0be9ad6177ae683a32a86204530e8ee624a0f               # training corpus (the revision we used)
bash scripts/train_decoder.sh                                       # Hydra; outputs in train_results/marlin_decoder
```

**DreaMS encoder.** `scripts/train_dreams_encoder.sh` fine-tunes the full DreaMS backbone with a MIST-style growing
fingerprint head (cosine loss), then writes the embedding cache. The released encoder also mixed in ICEBERG-simulated
spectra (30% real examples per epoch): FRIGID's ICEBERG augmentation molecules re-simulated as raw peak lists with
ms-pred, so no formula reaches the encoder (`marlin.prep_iceberg_aug` → ms-pred `predict_smis.py --sparse-out` →
`marlin.iceberg_h5_to_mgf`). Train in float32: the embedding cache is float32 and a head trained under bf16 does not
transfer to it.

**MIST-CF formulas (MIST encoder only).** MIST-CF was retrained on the training split of `canopus_hplus_100_0` with
the [MIST-CF](https://github.com/samgoldman97/mist-cf) code (2 layers, hidden size 128, absolute-sine formula
encoder, learning rate 4.5e-4 with decay 0.88, batch size 4, up to 32 decoys, best of 200 epochs on validation).
Candidate formulas for each spectrum come from SIRIUS decomposition of the precursor ([M+H]+, 10 ppm, common
elements), reduced to 256 by MIST-CF's fast filter. The retrained model scores them, and the top five per spectrum
are stored in `top5_formulas_natural.json`. The MIST encoder's peak subformulae for the top-1 formula were then
assigned with MIST's `assign_subformulae.py` (vendored in `src/mist/subformulae/`; `--mass-diff-thresh 20
--max-formulae 50`).

## License

Code: Apache License 2.0 (`LICENSE`). `src/dlm` and `src/mist` contain the parts of FRIGID's code (Apache-2.0; itself
adapted from NVIDIA GenMol and MIST, MIT) that MARLIN uses; third-party notices are in `NOTICE` and
`third_party_licenses/`.

Model weights and derived data: CC BY-NC 4.0. The MARLIN decoder is fine-tuned from FRIGID's decoder checkpoint, which
is released under CC BY-NC 4.0; the DreaMS encoder is fine-tuned from the DreaMS weights (CC BY 4.0).

## Citation

```bibtex
@misc{che2026marlin,
  title         = {{MARLIN}: De Novo Molecular Structure Elucidation from Tandem Mass Spectra without a Ground-Truth Formula},
  author        = {Che, Xujun and Du, Xiuxia and Xu, Depeng},
  year          = {2026},
  eprint        = {2607.04774},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2607.04774}
}
```
