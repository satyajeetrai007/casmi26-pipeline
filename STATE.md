# State (update after every step)

## Public leaderboard
| when (UTC) | submission | public | notes |
|---|---|---|---|
| 2026-10-21 19:48 | V0 private fork of a public pipeline (reference only, never to be resubmitted) | 0.456 | |
| 2026-10-25 17:11 | V1 own pipeline: fp_v2 + PubChem window (top-500 by popularity) + GLACIER/ICEBERG + leak-free LambdaRank (kernel casmi26-final v2) | **0.287** | val LB estimate 0.599 -> ratio 0.48: validation was badly miscalibrated |

Public best at hand-over: 0.467 (public notebooks), LB #1 0.504.

## GPU ledger (owner: 19 h available at 2026-10-25 17:00 UTC)
| kernel / version | started (UTC) | ended (UTC) | GPU h | notes |
|---|---|---|---|---|
| casmi26-final v2 | 2026-10-25 16:03 | 17:09 | 1.1 | V1 submission notebook (65 min) |
| casmi26-train-xenc v1 | 17:29 | 17:53 | 0.4 | crashed at init (self.type name clash) - fixed |
| casmi26-train-xenc v2 | 17:54 | running | <= 5.5 | cross-encoder xenc_v1, max_hours 5.3, 8 epochs x 5000 steps |
Used so far: ~1.5 h. Remaining: ~17.5 h (keep 2 h reserve for final kernels).

## Running at hand-over (2026-10-25 18:20 UTC) — the hourly routine owns the queue from now on
- casmi26-train-xenc v2 (GPU) — when COMPLETE: read its log (per-epoch `[epN EMA] MRR@25 on truth + same-formula lists`, columns xenc vs fp_ll) and follow queue item 1 in CLOUD_AGENT.md.
- V1 submission 57637669 (casmi26-final v2) — public score pending; record it above when it appears and compute the ratio (LB / val estimate 0.599).
- The owner's PC may also be on with read-only watchers; it will not push kernels or submit.

## Model results (validation, class-2 MRR@25 unless noted)
| model | S1 | S2 | S3 | notes |
|---|---|---|---|---|
| fp_v1 run 3 (12 ep) fused | 0.505 | 0.807 | 0.592 | fp head alone S1 0.517 |
| fp_v2 (fine-tune, NP x3) fused | 0.521 | 0.808 | 0.600 | fp head S1 0.527 |
| ranker v5 (fp_v2, + GL/ICE, leak-free) | 0.965 (c2) | 0.973 | 0.808 | LB estimate 0.599; priors-only S3 0.37 |

## Diagnosis of V1 (2026-10-25 18:30 UTC) and calibration chain (driven from the owner's PC, do NOT push these kernels)
Hypotheses: (1) the PubChem channel kept the 500 MOST DOCUMENTED window members, so poorly documented class-2 answers never got scored;
(2) the ranker leans on PubChem popularity, which validation (documented library structures, always in the pool) rewards but class-2 test answers (no public spectra) do not.
Honest class-2 validation: `infer_fp.py --class2 --truth_pubchem` removes the truth from the pool, so it can only re-enter through the PubChem window (as at test time).
New PubChem selection `--pc_select fp`: pre-score up to 20,000 window members by fingerprint log-likelihood, keep the best 400 + the 100 most documented.
| kernel | device | what |
|---|---|---|
| casmi26-infer-hc-a | CPU | honest class-2 records, V1 popularity selection (`val_c2tp_pop.pkl`) |
| casmi26-infer-hc-b | CPU | honest class-2 records, spectral-fit selection (`val_c2tp_fp.pkl`) |
| casmi26-infer-hc-c | CPU | class-1 records + test records, spectral-fit selection (`val_c1_fp.pkl`, `test_scores_fp.pkl`) |
| casmi26-gl-hc-a / gl-hc-bc / gl-hc-c | GPU ~40 min each | GLACIER/ICEBERG on each record file separately (per-file, to avoid a 'missing simulator score = truth' leak) |
| casmi26-rank-cal | CPU | frozen V1 ranker on V1 class-1 + honest class-2 (pop) -> its LB estimate should be ~0.29 if the diagnosis is right |
| casmi26-rank-hc | CPU | new ranker CV on class-1 fp + honest class-2 fp -> honest LB estimate of the new configuration |
**Submission rule from now on:** only when rank-cal reproduces V1 (estimate within ~0.05 of 0.287) AND the new configuration's honest estimate exceeds 0.48 (clearly above the 0.456/0.467 public line).
