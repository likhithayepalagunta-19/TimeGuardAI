# Initial benchmark results

Measured 19 October 2026, seed 42, Python 3.12.14, scikit-learn 1.9.1. CPU: Intel64 Family 6 Model 186 Stepping 3, GenuineIntel. Data archive hashes are recorded in `reports/data_provenance.json`. Exact splits are retained in the locally generated model artifacts. All values come from `reports/rul.json` and `reports/modes.json`.

## NASA C-MAPSS FD001

60 fitting engines, 20 calibration engines and 20 validation engines. Quantile models train on RUL capped at 125 cycles. Conformal correction uses raw RUL at one selected prefix endpoint per calibration engine, sampled at 60–95% of lifetime. Official test evaluation uses 100 final endpoints, no short-engine exclusions. The calibration-prefix distribution may differ from test truncation.

| Measure | Squared-error baseline | Quantile median | Conservative τ=.20 decision |
|---|---:|---:|---:|
| Clipped-label RMSE, cycles | 13.05 | 14.08 | 15.81 |
| Raw-label RMSE, cycles | 14.21 | 14.90 | 17.62 |
| NASA S-score, raw labels | 308.42 | 398.78 | 418.71 |
| Any positive RUL overestimate | 62% | 65% | 28% |

The conservative estimate reduces the late-estimate rate but worsens RMSE and NASA score. A lower late rate alone does not demonstrate better maintenance economics. The squared-error baseline is stronger on these accuracy metrics and must remain visible.

Nominal 90% intervals achieve **84/100 = 84%** official-test empirical coverage, with **54.27 cycles** mean width. Held-out validation coverage is 90%. The correction is 3.30 cycles. The official test coverage target is **not met**. Do not change calibration based on the revealed test outcomes and claim an untouched test. Further calibration research needs fresh validation or a preregistered independent protocol.

Warm model-only inference for all four RUL estimators, one row per call, 10 warmups and 100 measured repetitions: **p50 4.35 ms**, **p95 7.92 ms**. This excludes feature extraction, SHAP, replay, network and UI time. It is not end-to-end latency.

## UCI AI4I

Binary-stratified random 80/20 split. 2,000 holdout rows, 68 binary positives. Generated-row dependence remains a limitation. All thresholds fixed at 0.5. Mode models share the same row split but are independent labels.

| Target | Precision | Recall | F1 | PR-AUC | Positive support |
|---|---:|---:|---:|---:|---:|
| Machine failure | .9492 | .8235 | .8819 | .8824 | 68 |
| TWF | 0 | 0 | 0 | .0724 | 10 |
| HDF | .9310 | .9310 | .9310 | .9455 | 29 |
| PWF | 1 | 1 | 1 | 1 | 13 |
| OSF | 1 | 1 | 1 | 1 | 16 |
| RNF | 0 | 0 | 0 | .0321 | 4 |

High scores for some labels partly reflect the synthetic generation rules and small positive supports. Tool-wear and random-failure recall are zero at the chosen threshold. Do not present a macro success claim that hides these failures. RNF has no meaningful physical predictor by construction. Explore thresholds only on a separate validation split in a future preregistered run.

## Not measured

Full maintenance cost simulation, industrial downtime reduction, alert-episode rate on held-out trajectories, full end-to-end latency, drift-conditioned coverage and explanation latency. TreeSHAP export script is provided but has not been executed. Streamlit source compiles, but only the independent Stage 1 browser portal was interactively inspected.
