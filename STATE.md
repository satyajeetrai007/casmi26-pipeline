# State (update after every step)

## Public leaderboard
| when (UTC) | submission | public | notes |
|---|---|---|---|
| 2026-10-21 19:48 | V0 private fork of a public pipeline (reference only, never to be resubmitted) | 0.456 | |
| 2026-10-25 ~17:40 | V1 own pipeline: fp_v2 + PubChem window + GLACIER/ICEBERG + leak-free LambdaRank (kernel casmi26-final v2) | pending | val LB estimate 0.599 |

Public best at hand-over: 0.467 (public notebooks), LB #1 0.504.

## GPU ledger (owner: 19 h available at 2026-10-25 17:00 UTC)
| kernel / version | started (UTC) | hours | notes |
|---|---|---|---|
| casmi26-final v2 | 16:03 | ~1.5 | V1 submission notebook |
| casmi26-train-xenc v1 | after molfeat | <= 5.5 | cross-encoder xenc_v1 (max_hours 5.3) |

## Running at hand-over
- casmi26-molfeat v1 (CPU) -> then casmi26-train-xenc v1 is pushed automatically by the local PC watcher (if the PC was off, push it yourself: `./kag.sh push k_train_xenc`).
- casmi26-final v2 -> submitted as V1 by the local watcher once complete (check `./kag.sh subs`).

## Model results (validation, class-2 MRR@25 unless noted)
| model | S1 | S2 | S3 | notes |
|---|---|---|---|---|
| fp_v1 run 3 (12 ep) fused | 0.505 | 0.807 | 0.592 | fp head alone S1 0.517 |
| fp_v2 (fine-tune, NP x3) fused | 0.521 | 0.808 | 0.600 | fp head S1 0.527 |
| ranker v5 (fp_v2, + GL/ICE, leak-free) | 0.965 (c2) | 0.973 | 0.808 | LB estimate 0.599; priors-only S3 0.37 |
