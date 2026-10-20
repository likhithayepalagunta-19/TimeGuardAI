# TimeGuard AI — project summary

A full-stack predictive maintenance portfolio project by Likhitha Yepalagunta.

**Portal:** https://time-guard-ai.vercel.app/

**API docs:** https://timeguard-ai-api.onrender.com/docs

**Health:** https://timeguard-ai-api.onrender.com/health

**Repository:** https://github.com/likhithayepalagunta-19/TimeGuardAI

## Project description

TimeGuard AI is a deployed predictive-maintenance workbench combining remaining-life forecasts, explicit uncertainty, local TreeSHAP explanations and persistent maintenance recommendations. Separate NASA C-MAPSS and UCI AI4I models address degradation trajectories and overlapping manufacturing failure modes. A Vercel portal connects to FastAPI on Render, with labeled recorded-model replay when the API is unavailable. Maintenance authority remains with the engineer.

## Implementation and measured results

Causal 30-cycle features feed quantile RUL models trained with labels capped at 125 cycles. On 100 official FD001 test endpoints, median raw RMSE is **14.90 cycles** and NASA S-Score is **398.78**. A conservative **tau=.20**, representing a **4:1 late-vs-early linear cost penalty**, reduces late overestimates from **65% to 28%**, with worse RMSE and NASA score; the squared-error baseline remains better on those accuracy metrics. Split conformal intervals have a nominal **90%** target but achieve **84% empirical test coverage**, an explicit calibration gap. AI4I binary precision/recall/F1 are **94.92%/82.35%/88.19%** on 2,000 holdout rows. TWF and RNF recall are zero at the default 0.5 threshold. Live scenario presets demonstrate a low baseline score, **HDF approximately 97.5%** and **OSF approximately 91.9%**; these individual probabilities are uncalibrated and do not establish holdout performance.

The portal supports live inference, local TreeSHAP, packet-loss holds and sensor-offset review. Fast Demo replays verified model outputs from cycle 80 to 180 in four seconds for presentation speed. Sensor names are display aliases with raw C-MAPSS IDs preserved, not validated physical metadata. CSV export is an inference and decision audit log for maintenance review, not an automatically dispatched work order. After dependency installation, `python evaluate.py` verifies manifest-listed artifacts and writes reproducible JSON/CSV evidence to `reports/`. Real factory savings, drift-conditioned coverage and end-to-end cloud latency remain unvalidated. See [system walkthrough](SYSTEM_WALKTHROUGH.md) and [benchmark details](BENCHMARK_RESULTS.md).
