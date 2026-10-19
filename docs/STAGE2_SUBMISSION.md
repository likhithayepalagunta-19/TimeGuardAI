# Final submission portal copy

## Project description

TimeGuard AI is a software workbench for maintenance decisions from industrial sensor streams. It combines cost-sensitive remaining-life forecasts, calibrated prediction intervals and interpretable sensor evidence with a live replay interface. A separate manufacturing diagnosis model identifies overlapping failure modes. The system aims to give operators an actionable maintenance window and an explicit review state when telemetry becomes unreliable.

## Solution overview, implementation and results: truthful current version

We built a pastel portal with synthetic telemetry replay, sensor-fault scenarios, persistent alerts and CSV export, plus separate C-MAPSS RUL and AI4I diagnosis pipelines. On 100 official FD001 test endpoints, median RUL RMSE was 14.90 raw cycles. A conservative decision quantile reduced late estimates from 65% for the median to 28%, with higher RMSE. Nominal-90% intervals achieved 84% test coverage with 54.27-cycle mean width, exposing a calibration gap. AI4I binary precision/recall/F1 were 94.92%/82.35%/88.19% on a 2,000-row holdout. Current TWF and RNF recall were zero at threshold 0.5. Warm model-only p95 latency for all four RUL estimators was 7.92 ms on the recorded Intel CPU. The squared-error RUL baseline performed better on RMSE and NASA score. TreeSHAP export code is included but unexecuted. Streamlit replay/diagnosis source is provided. Industrial savings remain unvalidated.

## Replace after the final benchmark run

We evaluated [N] official FD001 test engines using one final endpoint per engine. Median RUL achieved [RMSE] cycles on [raw/clipped] labels and NASA S-score [score] on raw labels. The conservative decision estimate changed late-estimate rate from [baseline]% to [ours]%. Nominal 90% intervals achieved [coverage]% empirical coverage with [width] cycles mean width. AI4I binary failure precision/recall/F1 were [P]/[R]/[F1], with per-mode support and RNF results in our report. Warm model-only p95 latency was [ms] on [CPU], excluding feature extraction, SHAP and UI. Under the disclosed simulated cost model, total decision cost changed by [percentage] versus [baseline]. These benchmark and simulation outcomes do not establish real factory downtime reduction.

Do not paste the replacement paragraph until every placeholder has evidence. If a component is unfinished, identify it as proposed. Add repository commit, demo/video links and model/data provenance required by the actual organizer form.
