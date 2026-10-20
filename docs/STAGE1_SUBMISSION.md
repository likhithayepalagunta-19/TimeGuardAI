# Stage 1 screening copy

> **Historical screening-stage copy.** This file preserves the Stage 1 proposal and its targets. Current deployed functionality, measured results and presentation disclaimers are in the [README](../README.md), [Stage 2 submission](STAGE2_SUBMISSION.md) and [Loom script](DEMO_SCRIPT.md). Earlier synthetic-demo descriptions and impact targets are not current deployment claims.


Dates supplied by the team: idea screening 19 October 2026, development/final judging 20 October 2026. Confirm organizer timing, allowed pre-event preparation, slide limits and form word limits. The supplied reference PDF is an unrelated SIH deck and does not establish this event's rules. The fields below follow the user's requested rubric, not an independently obtained official form.

## Team name and members

Team name: [Insert registered team name]

Members: [Full name — ML and evaluation], [Full name — data and streaming], [Full name — interface], [Full name — integration and pitch]

## Selected problem statement

Track 1 — AI/ML Methodology. Challenge 10: TimeGuard AI — Predicting Machine Failure from Sensor Streams. Difficulty: Intermediate to Advanced.

## Problem understanding

Industrial maintenance is a decision problem under unequal error costs. Overestimating remaining useful life can delay maintenance until an outage, while repeated premature alerts waste component life and operator attention. Sensor noise, operating-regime changes, missing packets and calibration drift can make a precise forecast unreliable. Maintenance teams need an actionable warning window, the measurements that influenced it, and an explicit indication when the model cannot support a trustworthy recommendation. A useful system must therefore evaluate warning timing and false-alert episodes as well as prediction accuracy.

## Proposed solution

We propose TimeGuard AI, a software maintenance workbench that replays telemetry as time-ordered packets and converts causal sensor features into remaining-life forecasts with uncertainty. A cost-sensitive decision estimate favors earlier intervention when late action is expensive. A conformal calibration layer provides a prediction interval whose empirical coverage we measure on held-out engines. The maintenance policy requires persistent evidence, suppresses repeated alerts and requests operator review when telemetry quality deteriorates. Local sensor attributions explain model behavior. A separate AI4I diagnosis workbench predicts overlapping manufacturing failure modes. The two domains share a dashboard, but remain statistically separate.

## Innovation and novelty

The contribution is the integration and evaluation of prediction, uncertainty and maintenance action. Standard regression can achieve a reasonable average error while warning too late on expensive failures. TimeGuard explicitly prices that asymmetry, measures interval coverage and width, and evaluates alert episodes and simulated maintenance cost. Controlled packet loss and sensor-offset scenarios make model trust visible during a live demo. The system abstains when input quality fails instead of displaying unsupported certainty. This is a distinctive engineering combination for the challenge, not a claim that quantile regression, conformal prediction or SHAP are new inventions.

## Proposed technology and AI approach

For NASA C-MAPSS FD001 we split by engine, apply causal EMA and extract 30-cycle trailing sensor means, standard deviations, min/max and slopes. Training labels use piecewise linear RUL capped at 125 cycles. Boosted trees estimate the 5th, 50th and 95th percentiles using pinball loss. An additional 20th-percentile model corresponds to a four-to-one late-versus-early linear cost assumption. Held-out calibration engine endpoints provide a finite-sample conformal correction to the outer quantiles. We report marginal empirical coverage and do not claim guaranteed coverage during drift. TreeSHAP explains the fitted RUL estimator. Separate balanced random forests predict AI4I binary failure and five non-exclusive mode labels from physical inputs, excluding all target and identity fields. A Streamlit replay presents the actual trained RUL model, while a dependency-free pastel browser portal supports Stage 1 synthetic demonstrations.

## Expected impact

We aim to improve maintenance lead time while controlling unnecessary interventions. Acceptance targets are a 20% reduction in simulated decision cost against a tuned threshold baseline, 90% nominal interval coverage with reported mean width, and model-only p95 inference below 100 ms on a declared CPU. We will also report NASA S-score, late-estimate rate, minority-mode recall and false-alert episodes per 1,000 cycles. These are targets, not achieved industrial outcomes. For illustration, avoiding ten four-hour outages at ₹50,000 per hour yields ₹20 lakh in gross annual downtime value before implementation and extra-maintenance costs. Plant-specific trials are required to validate the assumptions.

## Brief implementation plan: 7.5-hour coding window

| Elapsed time | Work | Exit evidence |
|---|---|---|
| 00:00–00:45 | Verify dataset files, provenance, label schemas and engine splits | Saved manifest and split IDs |
| 00:45–01:30 | Implement causal features and baseline | Prefix invariance check and validation RMSE |
| 01:30–02:30 | Train quantile RUL models and conformal calibration | Held-out coverage and width |
| 02:30–03:15 | Train separate AI4I failure/mode models | Per-label support, precision/recall and PR-AUC |
| 03:15–04:00 | Connect replay, persistence policy and quality gate | Real-prefix inference and alert log |
| 04:00–04:45 | Add TreeSHAP, evidence export and pastel interface polish | Explanation from a fitted model |
| 04:45–05:30 | Run locked test evaluation and ablation | Saved metrics and environment record |
| 05:30–06:00 | Stabilize and record the core demo | Reproducible offline demo and walkthrough |
| 06:00–06:45 | Rehearse failure/drift scenarios, improve accessibility | Clear fallback and operator wording |
| 06:45–07:30 | Freeze code, update README, record final video and submit | Correct repository/deck links and form receipt |

Core prototype is targeted within six hours. The remaining 1.5 hours protect reproducibility and submission quality. If schedule slips, defer the temporal neural network and PRONOSTIA, retain real RUL evaluation and honest uncertainty reporting.

## 45-second opening

“A factory does not need another health score. It needs enough warning to plan a repair and a reason to trust that warning. TimeGuard forecasts remaining life as a range, accounts for the higher cost of late intervention, and shows which sensor measurements drove the estimate. Watch the same machine under clean telemetry and a sensor fault: the system changes its maintenance recommendation and asks for review when its inputs become unreliable. Our benchmark plan evaluates the full decision, including false alerts and maintenance cost, rather than accuracy alone.”
