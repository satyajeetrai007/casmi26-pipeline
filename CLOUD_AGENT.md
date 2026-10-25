# CASMI 2026 (Kaggle) — runbook for the cloud agent

You are continuing an own-model pipeline for the Kaggle competition **Enveda CASMI 2026 – Molecule ID From Mass Spectra**
(`enveda-CASMI26-molecule-id-mass-spectra`, MRR@25 on InChIKey-14 after RDKit tautomer canonicalisation, final deadline 2026-12-14).
All compute runs as **Kaggle kernels** under the account `satyajeetrai`; this machine only builds, pushes and reads them.
Read `STATE.md` (GPU ledger, submissions, what is running) and the end of `EXPERIMENTS.md` first, every time.

## Setup (each fresh cloud session)
```bash
pip install -q kaggle            # Kaggle CLI 2.x
export KAGGLE_API_TOKEN=...      # provided as an environment secret; never print it, never commit it
python make_kernels.py .         # regenerates the k_* kernel folders from colab/*.py (k_* is git-ignored)
chmod +x kag.sh && ./kag.sh subs # smoke test: lists our submissions
```
Helper: `./kag.sh push k_<dir>` (retries while both GPU slots are busy) · `./kag.sh wait casmi26-<slug>` · `./kag.sh logs casmi26-<slug> 80`
· `./kag.sh output casmi26-<slug> out/<dir> '<regex>'` · `./kag.sh submit casmi26-final <version> "<msg>"` · `./kag.sh subs`.

## Owner's standing rules (do not break)
1. **Never copy, fork or resubmit other participants' public notebooks.** Public notebooks/discussions are for ideas and LB calibration only. Papers and permitted public data/models are fine.
2. Only our own trained models or reproducible third-party models (GLACIER/ICEBERG MassSpecGym weights, DreaMS, MIST are allowed). **NIST and anything NIST-derived is forbidden.** Shared participant weights without public training code are not prize-eligible: do not use them.
3. **Submissions**: you may submit without asking, but only a version whose validation estimate beats our best public score by a clear margin (not a weight tweak). Max 5 per day; leave at least 2 per day unused. Record every submission in `STATE.md`.
4. Do not tune approaches that are far below our best; kill them and try the next structural idea.
5. GPU: the owner had **19 GPU-hours at 2026-10-25 17:00 UTC**. Log every GPU kernel's runtime in `STATE.md`; keep a 2 h reserve for the final submission kernel. CPU kernels are free; prefer them for anything that does not need a GPU.
6. Commit and push `STATE.md` / `EXPERIMENTS.md` after every step, so the owner can follow from GitHub.

## Kernel graph (slug = `casmi26-<name>`, folder = `k_<name>`)
| kernel | device | does | inputs |
|---|---|---|---|
| cpu-prep-cache-pools-baseline | CPU | peak cache of train.parquet (electron-mass fix), 1.14M candidate pool (train+COCONUT+PubChemLite+ChEBI/LMSD, re-keyed by RDKit InChIKey-14), validation sets, cosine baseline | competition data, public candidate datasets |
| pop | CPU | joins PubChem documentation counts onto the pool | prep, dmitriigluzdov/casmi26-pubchem-popularity-prior (data only) |
| train-fp | GPU | own spectrum transformer (fingerprint + contrastive heads); latest = `runs/fp_v2` | prep |
| molfeat | CPU | atoms / cleavage-fragment masses / count FP of every pool structure | prep |
| train-xenc | GPU | **spectrum x candidate cross-encoder** (`colab/xenc.py`, `train_xenc.py`), hard same-formula negatives | prep, molfeat, dataset satyajeetrai/casmi26-fp-v2-ema |
| infer-fp | CPU | candidate records for validation (class-1 and class-2 modes) and test: pool + 500 PubChem-window candidates, fingerprint/contrastive/library/analog/fragment features | prep, train-fp, pop |
| gl-val | GPU | GLACIER + ICEBERG (MassSpecGym weights) on the top-40 of each list, val and test | infer-fp |
| xenc-val | GPU | cross-encoder score for every validation candidate | infer-fp, train-xenc |
| rank | CPU | leak-free LambdaRank, 5-fold CV by molecule, prints the LB estimate | infer-fp, gl-val, (xenc-val) |
| final | GPU | **the submission notebook**: test inference + simulators + (xenc) + ranker at run time | prep, train-fp, pop, rank, (train-xenc) |
Build with `XENC=1 python make_kernels.py .` to attach the cross-encoder to `rank` and `final`.

## LB estimate (the number that decides a submission)
Use **estimate C = 0.16 * MRR(S1|c1) + 0.55 * mean(MRR(S2|c2), MRR(S3|c2))** computed on HONEST records (`--class2 --truth_pubchem`); it reproduced V1 exactly (0.287). The old estimate (S3 low+mid popularity, truth in pool) over-estimated V1 by 2x and must not be used.
S1 = 250 timsTOF natural products (class-1 behaviour), S3 = natural-product-library structures with all their spectra removed (class-2 behaviour), novel molecules (~29%) score ~0.
Known calibration: the public LB moves far less than validation; first own submission (V1) will calibrate this — compare its LB with its estimate and write the ratio into `STATE.md`.

## Experiment queue (top first; update this list as results come in)
0. **V1 scored 0.287 against a validation estimate of 0.599.** Read "Diagnosis of V1" in `STATE.md`. Until `casmi26-rank-cal` reproduces V1's score (estimate within ~0.05 of 0.287), no validation number is trusted and nothing is submitted. The calibration chain (infer-hc-*, gl-hc-*, rank-cal, rank-hc) is driven from the owner's PC; record its results when the kernels finish.
1. If calibrated: compare `rank-hc` (spectral-fit PubChem selection, ranker trained on honest records) with rank-cal. If its honest estimate > 0.48, build a `final` variant with `--pc_select fp` + the rank-hc ranker, check the commit run, submit once.
2. The cross-encoder (xenc) is a **dead end** (lost to the fingerprint score by 0.1-0.25 MRR). Do not resume or tune it.
3. Next structural ideas, in order: (a) more PubChem recall (pre-score the whole window, larger --pubchem_top); (b) spectrum-model ensembling (second seed / fp_v1 + fp_v2 log-lik averaged); (c) fine-tune GLACIER on timsTOF training spectra (large effort).

## Pitfalls already paid for (read before changing features)
- Validation truths are always **training structures**; class-2 test answers never are. Any feature revealing train membership leaks: pool source flags, PubChem depositor bit 1 (spectral library), the COCONUT membership bit (the public COCONUT file excludes train structures). They are removed; keep them out.
- The visible `test.parquet` is copied from training spectra, so the commit run of `final` always ranks training structures first — expected, not a leak. The scoring rerun swaps in the hidden test (new molecule ids), so everything must be computed at run time inside `final`.
- `prvsiyan/coconut-casmi26-candidates` keys are not InChIKey-14s: always recompute keys with RDKit.
- Kaggle images are Python 3.13 without RDKit: kernels install `metric/rdkit-2026-3-3-wheel` offline; the simulators need RDKit 2025.3.6 cp313 from `eveliaveldrine/rdkit-wheels`.
- T4 memory: spectrum transformer bs 512 needs both T4s (DataParallel).
- `kaggle kernels logs` only works after a run ends; `push` fails while two GPU sessions run (the helper retries).

## How each hourly run works (routine)
1. `git fetch origin`; if branch `agent-state` exists, work on it (`git checkout agent-state`), otherwise on `main`.
2. Read this file, `STATE.md`, the end of `EXPERIMENTS.md`. Setup as above; verify `./kag.sh subs`.
3. Poll every kernel listed as running in `STATE.md`; for finished ones read the log, record results (and GPU hours) and decide by the queue.
4. Take only the next step(s) that do not depend on unfinished work. Never wait more than ~5 minutes in one run; never train on the cloud machine (no GPU).
5. Commit what changed and push: `git push origin HEAD:main`, or `git push origin HEAD:agent-state` if main is rejected.
6. If blocked (auth, quota, a crash you cannot fix confidently), write it under **Blocked** in `STATE.md`, commit, push and stop.
