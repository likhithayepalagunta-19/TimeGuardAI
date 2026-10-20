# System design notes

Design rationale, model choices and proposed operational validation for TimeGuard AI. See the [README](../README.md) for deployed behavior.

## Problem understanding

Industrial maintenance is a decision problem under unequal error costs. Overestimating remaining useful life can delay maintenance until an outage, while repeated premature alerts waste component life and operator attention. Sensor noise, operating-regime changes, missing packets and calibration drift can make a precise forecast unreliable. Maintenance teams need an actionable warning window, the measurements that influenced it, and an explicit indication when the model cannot support a trustworthy recommendation. A useful system must therefore evaluate warning timing and false-alert episodes as well as prediction accuracy.

## System design

TimeGuard AI is a software maintenance workbench that replays telemetry as time-ordered packets and converts causal sensor features into remaining-life forecasts with uncertainty. A cost-sensitive decision estimate favors earlier intervention when late action is expensive. A conformal calibration layer provides a prediction interval whose empirical coverage is measured on held-out engines. The maintenance policy requires persistent evidence, suppresses repeated alerts and requests operator review when telemetry quality deteriorates. Local sensor attributions explain model behavior. A separate AI4I diagnosis workbench predicts overlapping manufacturing failure modes. The two domains share a dashboard, but remain statistically separate.

## Integration rationale

The contribution is the integration and evaluation of prediction, uncertainty and maintenance action. Standard regression can achieve a reasonable average error while warning too late on expensive failures. TimeGuard explicitly prices that asymmetry, measures interval coverage and width, and defines evaluation extensions for alert episodes and simulated maintenance cost; those measurements are not yet implemented. Controlled packet loss and sensor-offset scenarios make model trust visible during a live demo. The system abstains when input quality fails instead of displaying unsupported certainty. The contribution is systems integration rather than a new quantile regression, conformal prediction or SHAP algorithm.

## Model and feature design

For NASA C-MAPSS FD001 the pipeline splits by engine, applies causal EMA and extracts 30-cycle trailing sensor means, standard deviations, min/max and slopes. Training labels use piecewise linear RUL capped at 125 cycles. Boosted trees estimate the 5th, 50th and 95th percentiles using pinball loss. An additional 20th-percentile model corresponds to a four-to-one late-versus-early linear cost assumption. Held-out calibration engine endpoints provide a finite-sample conformal correction to the outer quantiles. The evaluation reports marginal empirical coverage without guaranteeing coverage during drift. TreeSHAP explains the fitted RUL estimator. Separate balanced random forests predict AI4I binary failure and five non-exclusive mode labels from physical inputs, excluding all target and identity fields. The deployed pastel portal and FastAPI service expose fitted-model inference; an optional Streamlit workbench is also available.

## Pilot acceptance criteria (unvalidated targets)

A future operational pilot should assess maintenance lead time and unnecessary interventions. Acceptance targets are a 20% reduction in simulated decision cost against a tuned threshold baseline, 90% nominal interval coverage with reported mean width, and model-only p95 inference below 100 ms on a declared CPU. The existing benchmark reports NASA S-score, late-estimate rate and minority-mode recall; false-alert episodes per 1,000 cycles remain a future measurement. These are targets, not achieved industrial outcomes. For illustration, avoiding ten four-hour outages at ₹50,000 per hour yields ₹20 lakh in gross annual downtime value before implementation and extra-maintenance costs. Plant-specific trials are required to validate the assumptions.

## Delivery sequence

| Phase | Engineering work | Exit evidence |
|---|---|---|
| Data contracts | Validate provenance, label schemas and engine splits | Manifest and split IDs |
| Causal modeling | Implement trailing features and accuracy baseline | Prefix invariance and validation metrics |
| Uncertainty | Fit quantile models and endpoint calibration | Coverage and interval width |
| Diagnostics | Fit separate binary and failure-mode classifiers | Per-label support and precision/recall |
| Inference integration | Connect replay, persistence policy and quality gates | Real-prefix inference and decision audit |
| Explainability and interface | Connect TreeSHAP, audit export and responsive views | Fitted-model attribution and browser checks |
| Reproducibility | Lock evaluation and publish artifacts | Parity checks and machine-readable reports |
| Deployment | Separate frontend and model service, exercise fallback | Readiness and provenance checks |

Factory savings, alert-episode reduction and simulated maintenance economics require independent evaluation. Current measured outcomes and limitations are in the [benchmark report](BENCHMARK_RESULTS.md).
