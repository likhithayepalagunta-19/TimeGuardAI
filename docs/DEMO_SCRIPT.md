# Three-minute jury demonstration

1. **0:00–0:25 — Frame the decision.** “Late maintenance is expensive. Constant alarms are expensive too. We forecast when to act and expose uncertainty.” Keep the synthetic demo label visible.
2. **0:25–1:05 — Replay normal degradation.** Select Engine A, press Start, choose 8× and watch cycle, RUL range, warning persistence and sensor evidence. Explain that one replay tick represents a cycle, not an actual second of engine service.
3. **1:05–1:40 — Inject a sensor offset.** Enable the offset control. The portal requests review and widens its illustrative band. Explain that a production drift monitor must distinguish operating changes from faults using validated models and delayed outcomes.
4. **1:40–2:05 — Show packet loss.** Enable missing packets. The system holds the last prediction, marks it stale and gates actionable recommendations. Reset returns to the same deterministic stream.
5. **2:05–2:35 — Explain and export.** Show the subsystem contribution bars, explicitly described as heuristic simulation evidence. Export the CSV and show the policy events. Actual fitted-model TreeSHAP is a separate Stage 2 output.
6. **2:35–3:00 — Close with validation.** “Our RUL and failure-mode datasets remain separate. We compare RMSE, late warnings, coverage, false-alert episodes and simulated decision cost. The repository includes measured benchmark results and their limitations. Portal outputs remain synthetic.”

## Likely jury questions

- **What is new?** The tested decision workflow and uncertainty/quality integration. No claim of inventing the component methods.
- **Why not only an LSTM?** A CPU-friendly quantile model leaves time for leakage-safe evaluation, calibration and a reliable demo. A small causal TCN is a controlled stretch ablation.
- **Can conformal guarantee safety?** No. Coverage is marginal under assumptions. Shift, dependent windows and mismatched endpoint distributions limit it. Quality gates and explicit abstention support honest use.
- **Why two datasets?** One validates time-to-failure, the other overlapping failure diagnoses. We do not infer an engine's fault class with a model trained on unrelated manufacturing snapshots.
- **Are these sensors root causes?** Attributions explain the model, not physical causality. An engineer confirms the diagnosis.
- **What if data download fails?** Show the labeled synthetic portal. Report the benchmark as not run, retain the real-data loader and document the failure.
- **What can operators do?** Inspect measurements, acknowledge a warning, and schedule a maintenance review. The prototype never shuts a machine down.
