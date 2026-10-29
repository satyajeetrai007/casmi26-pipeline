# State (update after every step)

## Public leaderboard
| when (UTC) | submission | public | notes |
|---|---|---|---|
| 2026-10-21 19:48 | V0 private fork of a public pipeline (reference only, never to be resubmitted) | 0.456 | |
| 2026-10-26 00:10 + 14:31 | V2: spectral-fit PubChem selection + ranker trained on honest records (casmi26-final-hc v1) - submitted TWICE by mistake (routine + PC watcher) | **0.255** | estimate C 0.498 -> estimator C is wrong too |
| 2026-10-26 16:17 | P1 probe (CPU): pool only, spectral fused score, library hits >= 0.8 first, no ranker/simulators | 0.226 | diagnostic |
| 2026-10-26 16:55 | P2 probe (CPU): P1 + 500 PubChem-window candidates chosen by popularity | 0.184 | PubChem candidates hurt when ranked by spectral score alone |
| 2026-10-26 17:41 | P3 probe (CPU): P2 + 0.15 x popularity in the score | 0.215 | popularity recovers part of it |
| 2026-10-28 15:11 | P4 probe: pool only, fingerprint log-lik only, lib gate 0.8 | 0.227 | = P1 fused (0.226): contrastive adds nothing |
| 2026-10-28 15:12 | P5 probe: pool only, contrastive only, lib gate 0.8 | 0.203 | |
| 2026-10-28 15:12 | P6 probe: pool only, popularity prior only (spectrum-blind), lib gate 0.8 | 0.170 | our spectral score adds only +0.057 over popularity |
| 2026-10-28 17:04 | P7 probe: P1 + own mass-shifted analog channel (w_ap 0.5) | **0.264** | +0.038 over P1: analog channel confirmed on LB |
| 2026-10-28 ~19:00 | P8 probe: P7 + training structures demoted when library match < 0.8 | 0.230 | dead end: weak-library-match answers are often training structures |
| 2026-10-29 14:23 | P9: P7 + GLACIER/ICEBERG same-formula re-score (top 40) + tautomer dedupe (casmi26-final-p9 v2) | pending | |
| 2026-10-29 14:29 | P10: P7 + generated derivatives of the top-6 analogs, no penalty | 0.232 | derivative channel hurts when mixed in |
| 2026-10-25 17:11 | V1 own pipeline: fp_v2 + PubChem window (top-500 by popularity) + GLACIER/ICEBERG + leak-free LambdaRank (kernel casmi26-final v2) | **0.287** | val LB estimate 0.599 -> ratio 0.48: validation was badly miscalibrated |

Public best at hand-over: 0.467 (public notebooks), LB #1 0.504.

## GPU ledger (owner: 19 h available at 2026-10-25 17:00 UTC)
| kernel / version | started (UTC) | ended (UTC) | GPU h | notes |
|---|---|---|---|---|
| casmi26-final v2 | 2026-10-25 16:03 | 17:09 | 1.1 | V1 submission notebook (65 min) |
| casmi26-train-xenc v1 | 17:29 | 17:53 | 0.4 | crashed at init (self.type name clash) - fixed |
| casmi26-train-xenc v2 | 17:54 | 21:50 | 3.9 | cross-encoder xenc_v1: LOST to fp log-lik (S1 0.40 vs 0.65) -> killed |
| casmi26-gl-hc-a | ~19:20 | ~20:00 | 0.6 | simulators on honest class-2 (pop) records |
| casmi26-gl-hc-c / gl-hc-bc | 20:12 / pending | | ~0.7 each | simulators on class-1 fp + test fp / honest class-2 fp |
Used so far: ~7.4 h (incl. running gl-hc). Remaining: ~11.6 h (keep 2 h reserve for final kernels).

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

## Calibration result (2026-10-25 22:30 UTC)
| configuration (honest records) | S1 class 1 | S2 class 2 | S3 class 2 | estimate C = 0.16*S1 + 0.55*mean(S2,S3) |
|---|---|---|---|---|
| V1 (popularity-based PubChem selection, V1 ranker) | 0.978 | 0.014 | 0.461 | 0.287 (actual 0.287) |
| V2 (spectral-fit PubChem selection, rank-hc ranker) | 0.891 | 0.771 | 0.522 | 0.498 |
Estimator C was chosen after seeing V1's score (one calibration point): use it for decisions, and re-check it against V2's public score.

## Diagnosis after V2 + probes (2026-10-27)
- Own spectral retrieval alone (P1) = 0.226; the ranker + simulators add only ~+0.06-0.10 on top. The public engine alone reaches ~0.35, so the core spectrum model is the weak part, not the ranker.
- Library-token suspicion REFUTED (casmi26-libtok): encoding S1 spectra as `enveda-180` instead of `enveda-np-examples` drops S1 pool-only MRR 0.519 -> 0.427 (S2/S3 unchanged). Test inference keeps the np-examples token.
- Adduct shift table and 10 ppm windows checked: correct.
- Running (CPU, not submitted): casmi26-probe4-ll / probe5-con / probe6-prior = pool-only test submissions ordered by fingerprint log-lik only / contrastive only / popularity prior only (all with library gate 0.8). Compared with P1 (fused, 0.226) they show which score carries signal on the hidden test. Submitting them needs the owner's OK.

## Blocked
- The hourly routine (trig_01KBQwDr1zjhQzNHjqMeUTjx) is PAUSED: its GitHub push fails with 403 "Resource not accessible by integration", so it could not record its V2 submission and the PC submitted V2 again. Do not re-enable until a test run can push.
