# State (update after every step)

## Public leaderboard
| when (UTC) | submission | public | notes |
|---|---|---|---|
| 2026-10-21 19:48 | V0 private fork of a public pipeline (reference only, never to be resubmitted) | 0.456 | |
| 2026-10-25 ~17:40 | V1 own pipeline: fp_v2 + PubChem window + GLACIER/ICEBERG + leak-free LambdaRank (kernel casmi26-final v2) | pending | val LB estimate 0.599 |

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
