# Experiment log (newest last; one entry per finished experiment)

- 2026-10-22 baseline harness: cosine library search S1 class-1 MRR 0.905; truth in candidate top-100 for 99.6% (S1/S2); median same-formula pool candidates 50-60.
- 2026-10-22..24 own spectrum transformer (39M params, fingerprint + contrastive heads): 2.7 epochs on 1 T4 -> 12 epochs on 2xT4 with vectorised batches (4x faster). Class-2 fused MRR S1 0.42 -> 0.505, S2 0.62 -> 0.807, S3 0.52 -> 0.592.
- 2026-10-25 GLACIER + ICEBERG re-ranking of same-formula groups: GL+ICE lambda 1 lifts class-2 S1 0.39 -> 0.53, S2 0.77 -> 0.83, S3 0.50 -> 0.57.
- 2026-10-25 PubChem window channel (+500 most-documented PubChem-only structures per molecule): needed because the public 0.464 ablation shows ~10% of answers are reachable only there; truth is inside the 10 ppm PubChem window for 88% of validation molecules.
- 2026-10-25 ranker leak audit: features revealing training-structure membership (pool source flags, PubChem spectral-library depositor bit, COCONUT bit) inflated CV to 0.99; removed. Leak-free ranker v5: LB estimate 0.599 (class-2 proxy 0.82, spectrum-blind priors-only 0.37).
- 2026-10-25 fp_v2 fine-tune (NP libraries x3, lr 1.5e-4, 4.3 h): small gain (S1 fp head 0.517 -> 0.527).
- 2026-10-25 started: cross-encoder xenc_v1 (spectrum tokens x candidate atoms + cleavage-fragment masses + count FP, hard negatives from same-formula TRAINING structures only).
- 2026-10-25 **V1 own pipeline = 0.287 public** (validation estimate 0.599). Diagnosis in STATE.md.
- 2026-10-25 honest class-2 records (truth removed from the pool, only reachable through the PubChem window, V1 popularity-based PubChem selection): fingerprint fused S3 0.21, +GLACIER/ICEBERG 0.26, S2 ~0.01 (vs 0.59-0.81 on the old validation) -> most class-2 answers never reach the candidate list.
- 2026-10-25 **cross-encoder xenc_v1: dead end.** 8 epochs, 3.9 GPU-h; on truth + same-formula lists S1 0.40 / S2 0.69 / S3 0.55 vs the fingerprint log-likelihood 0.65 / 0.82 / 0.63. Killed, not tuned.
- 2026-10-26 **V2 = 0.255 public** (estimate C 0.498): spectral-fit PubChem selection did not help on the hidden test; validation has now failed twice.
- 2026-10-26 CPU diagnostic probes: P1 pool-only spectral 0.226, P2 +PubChem by popularity 0.184, P3 +popularity weight 0.215. Core spectral retrieval is the bottleneck.

