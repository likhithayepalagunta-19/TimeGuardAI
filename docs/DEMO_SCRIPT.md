# TimeGuard AI — Team Un4seen | 3:30 Loom demo

AI Innovation Hackathon 2026 · Track 1 — AI/ML Methodology · Challenge 10: TimeGuard AI — Predicting Machine Failure from Sensor Streams.

- [Live portal](https://time-guard-ai.vercel.app/)
- [Live API Swagger documentation](https://timeguard-ai-api.onrender.com/docs)
- [Health endpoint](https://timeguard-ai-api.onrender.com/health)
- [Repository](https://github.com/likhithayepalagunta-19/TimeGuardAI)

## 0:00–0:45 — Hook and problem

**0:00–0:20 · Screen:** Open the live Turbofan Monitor on engine 20 at cycle 80, with the connected status and populated chart visible.

**Say:** “A machine rarely fails without warning. The challenge is knowing whether those warnings justify action. A late forecast risks downtime. An early alarm wastes component life. Repeated false alarms teach operators to ignore the system.”

**0:20–0:45 · Screen:** Point to median RUL, conformal interval and maintenance recommendation.

**Say:** “We’re Team Un4seen, presenting TimeGuard AI for Track One, Challenge Ten. Our workbench answers three questions: how much operating life remains, how uncertain is that estimate, and what should a maintenance engineer investigate? We combine forecasts, explanations, and a persistent decision policy in one deployed interface.”

## 0:45–1:30 — Architecture and methodology

**0:45–1:05 · Screen:** Show the README architecture diagram; trace frontend → API → models → policy.

**Say:** “Our Vercel frontend connects to a FastAPI backend on Render. Two separate benchmarks ground the system: NASA C-MAPSS for turbofan degradation, and UCI AI4I for manufacturing failure modes. They share a workbench, while retaining separate models and evaluation protocols.”

**1:05–1:30 · Screen:** Highlight the methodology, conservative quantile and benchmark coverage.

**Say:** “Causal thirty-cycle features feed quantile regressors trained with RUL capped at 125 cycles. A conservative quantile represents a four-to-one late-versus-early error cost. Split conformal calibration adds uncertainty bands; TreeSHAP explains sensor contributions. Median performance is 14.90 raw-cycle RMSE and 398.78 NASA score. Nominal ninety-percent intervals achieved eighty-four-percent test coverage—a limitation we explicitly report.”

## 1:30–3:00 — Portal demonstration

**1:30–1:45 · Screen:** Return to the portal, point to Live Backend Connected, click **⚡ Fast Demo Play**, and leave the pointer still for four seconds.

**Say:** “Here is the deployed portal connected to our backend. For presentation speed, Fast Demo replays verified model outputs from cycle eighty to one-eighty in four seconds. Watch the forecast, uncertainty band, and status change together.”

**1:45–2:05 · Screen:** Highlight rose critical cards, the curve and the decision log.

**Say:** “The forecast falls into the critical range, while the decision log records the maintenance transition. Planning requires three lower-bound crossings; recovery requires five safe cycles above a margin. That persistence is designed to reduce alert flicker. Our conservative estimate reduces benchmark overestimates from sixty-five to twenty-eight percent, with an accuracy trade-off.”

**2:05–2:20 · Screen:** Show Sensor attribution, hover a display alias to expose its raw sensor ID, and click **Export CSV**.

**Say:** “These TreeSHAP bars show which sensor features push remaining life up or down. Display names are presentation aliases; raw IDs remain available. This explains model behavior, rather than proving physical causation. One click exports the prediction and decision audit for maintenance review.”

**2:20–2:35 · Screen:** Enable packet loss and step through cycles 181, 182 and 183. Point to **STALE / Review** and the observed-cycle indicator at the lost tick. Disable packet loss.

**Say:** “Now we interrupt the stream. On a lost packet, the portal holds the last valid forecast and requests review. It exposes stale evidence instead of presenting it as a fresh measurement. Injected conditions also fall outside our clean-benchmark coverage claim.”

**2:35–3:00 · Screen:** Open Manufacturing Lab. Click **Safe Baseline**, **Trigger Heat Dissipation (HDF)**, and **Trigger Overstrain / Tool Wear (OSF)**. Wait for each live result.

**Say:** “In the separate manufacturing lab, each preset updates the telemetry and submits live inference. The baseline produces a low failure score. The heat scenario produces approximately ninety-seven-percent HDF probability; the overstrain scenario produces approximately ninety-two-percent OSF probability. Modes can overlap. These are uncalibrated model scores, and tool-wear and random-failure detection remain documented weaknesses.”

## 3:00–3:30 — Reproducibility, impact and closing

**3:00–3:15 · Screen:** Show a completed `python evaluate.py` terminal run, then briefly show the API docs.

**Say:** “Judges can reproduce our benchmark with one command: python evaluate dot py, after dependency installation. It verifies artifacts and saves JSON and CSV evidence. Our API is documented, and recorded replay keeps the presentation usable when the backend is unavailable.”

**3:15–3:30 · Screen:** Return to the completed degradation curve.

**Say:** “Industrial value must be validated in a pilot. Illustratively, avoiding two downtime hours at five thousand dollars per hour protects ten thousand dollars before maintenance costs. TimeGuard makes that decision traceable: a forecast, an uncertainty range, and an auditable reason to act. Thank you.”

## Recording checklist and claim boundaries

- Warm the API first. Keep portal, repository and API docs tabs ready, plus a completed evaluation terminal. Speak around 145 words per minute; pause for playback and live responses.
- For exact on-screen preset figures, HDF is approximately **97.5%**, OSF **91.9%**, and baseline binary failure **0.0%** for the documented inputs. TWF is approximately 21.9% on the overstrain input; do not imply it exceeds 85%.
- Fast Demo is recorded real-model playback, not live inference on every tick. Default history through cycle 80 is also recorded; connected regular stepping performs live inference.
- The nominal 90% interval target achieved 84% empirical test coverage: a six-percentage-point empirical calibration gap. It is not machine-specific certainty or validated drift coverage.
- The conservative tau=.20 model represents a 4:1 late-vs-early linear cost penalty. Late overestimates decrease from 65% to 28%, while RMSE and NASA score worsen. The baseline remains stronger on those accuracy metrics.
- TreeSHAP aliases are not validated physical sensor metadata; raw IDs are preserved. Explanations are not causal diagnoses.
- CSV export is an inference and decision audit log for maintenance review. No work order is dispatched and no equipment is controlled.
- TWF/RNF recall is zero at threshold 0.5 on the holdout. Scenario probabilities are uncalibrated and are not recall or accuracy.
- The ROI example is illustrative arithmetic, not measured factory savings. End-to-end cloud latency, uptime and drift robustness are not established.

## Judge verification

With Python 3.12 and an activated environment:

```bash
pip install -r requirements.txt
python evaluate.py
```

The evaluator verifies all manifest-listed model/data/reference artifacts, checks benchmark parity and writes JSON/CSV evidence to `reports/`. See [benchmark results](BENCHMARK_RESULTS.md) and [validation record](VALIDATION.md).
