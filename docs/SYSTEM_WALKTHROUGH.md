# System walkthrough

This guide describes the operator workflow and the provenance of each displayed result.

## Connect to the system

1. Open the [web portal](https://time-guard-ai.vercel.app/).
2. Open the top-right connection/settings pill. Select **Live ML Backend (Render API)** and enter `https://timeguard-ai-api.onrender.com` if needed.
3. Select **Connect / retry**. A green status indicates a verified model backend. Check [health](https://timeguard-ai-api.onrender.com/health) or inspect [Swagger documentation](https://timeguard-ai-api.onrender.com/docs) when diagnosing connectivity.

When the API times out, the portal falls back to labeled recorded model outputs. Retry after the service wakes. Recorded replay remains usable without live inference; it cannot score arbitrary new telemetry.

## Turbofan Stream Monitor

### Inspect the initial forecast

The default view selects FD001 engine 20 with historical predictions through **cycle 80**, including median RUL, the shaded conformal interval and the conservative estimate. Historical points are verified recorded model outputs. With a connected backend, the current forecast and regular cycle stepping use live inference.

RUL is measured in **cycles**, not hours. The median supports accuracy assessment; the tau=.20 estimate represents a 4:1 late-vs-early linear cost assumption. The nominal 90% interval achieved 84% empirical coverage on 100 official test endpoints. It is not a 90% guarantee for the selected machine.

### Advance the stream

- **Start replay** advances the selected engine at the chosen speed; **Step** advances one cycle.
- **Reset** restores history through cycle 80, or the available final cycle for a shorter engine.
- **⚡ Fast Demo Play** selects engine 20, clears injected conditions, sets the maintenance horizon to 45 cycles and replays verified outputs from **cycle 80 to 180 in four seconds**. This is recorded playback, not 100 API inferences. Progress and attribution labels identify that source.

Metric accents follow median predicted RUL: mint/HEALTHY above 60, amber/INSPECTION DUE from 30 to 60, and rose/CRITICAL below 30. Critical tags pulse unless reduced motion is requested. Tags are advisory; no work order is automatically dispatched.

The maintenance policy is separate from the card color: three valid lower-bound crossings request planning; five valid safe cycles above the recovery margin clear it. The default recorded engine-20 scenario logs a planning transition at cycle 145. Policy state is reconstructed from the observed prefix and remains isolated between requests.

### Interpret sensor evidence

TreeSHAP shows signed contributions to the fitted median model in RUL cycles. Negative contributions reduce predicted life; positive contributions increase it. Temporal features are aggregated back to sensor IDs.

Names such as “HPT Coolant Bleed” and “HPC Outlet Pressure” are **presentation aliases**, not validated C-MAPSS physical mappings. Hover labels for raw IDs; attribution payloads also retain IDs. Consult the [reference sensor schema](TECHNICAL_DOSSIER.md). Model attribution is not proof of physical causality or fault diagnosis.

### Test telemetry quality

**Packet loss:** enable the control and advance the stream. Every third cycle is missing. From cycle 180, step through 181, 182 and 183 to observe a lost tick at 183. The portal holds the last valid forecast, exposes the observed cycle and requests review. The live path causally forward-fills missing inputs; offline replay only simulates delivery loss on clean recorded outputs.

**Sensor offset:** enable an s4 offset and choose its magnitude. With live inference, the offset applies to the last five supplied observations and triggers shift review. Clean-benchmark calibration does not validate coverage for this shifted scenario. Offline offsets are explicitly unscored; clean recorded predictions remain visible.

Disable injected conditions before comparing with the nominal benchmark. No robustness guarantee is implied by a successful stress-test interaction.

### Export an audit

Select **Export CSV** to download inference records and decision events for maintenance review. Records include cycle, observed cycle, model version, inference source, median/decision RUL, interval bounds, alert state, quality, injected conditions and attribution method. Export does not create a CMMS work order or control machinery.

## Manufacturing Diagnostics Lab

This view uses a separate AI4I model, not a turbofan failure classifier. Product type and five process measurements produce a binary failure score and independent TWF/HDF/PWF/OSF/RNF probabilities. Multiple modes can be active.

1. Open **Manufacturing lab**.
2. Choose a quick-test chip. It updates product type and all sliders, then immediately calls `/api/diagnose/ai4i` when connected.
3. Wait for the live result before comparing scores. Changing sliders clears previous scores; use **Analyze telemetry** to evaluate manual edits.

| Scenario | Type | Air / process temperature (K) | Speed (rpm) | Torque (Nm) | Wear (min) | Expected model output |
|---|---|---|---:|---:|---:|---|
| Safe Baseline | L | 300 / 310 | 1500 | 40 | 50 | Binary probability approximately 0.0% |
| Trigger Heat Dissipation (HDF) | L | 302 / 310 | 1350 | 50 | 50 | HDF approximately 97.5% |
| Trigger Overstrain / Tool Wear (OSF) | L | 300 / 310 | 1500 | 85 | 250 | OSF approximately 91.9%; TWF approximately 21.9% |

These are uncalibrated probabilities for specific inputs, not accuracy or recall measurements. TWF and RNF holdout recall are **zero at threshold 0.5**. High OSF does not imply high TWF; low scores do not establish safe operation. In offline mode, only exact recorded benchmark presets have supported predictions; custom telemetry requires the API.

## Verify results independently

With Python 3.12 and an activated environment:

```bash
pip install -r requirements.txt
python evaluate.py
```

The evaluator verifies manifest-listed artifacts, recomputes the frozen holdout metrics, checks parity and saves JSON/CSV evidence to `reports/`. Median raw RMSE is **14.90 cycles**, NASA S-Score **398.78**, and empirical interval coverage **84%**. Conservative decisions reduce late overestimates from **65% to 28%**, with higher RMSE and NASA score. See the [benchmark report](BENCHMARK_RESULTS.md) and [validation record](VALIDATION.md).

## Operational scope

The service is a public research workbench with software telemetry replay and typed prefix ingestion. Authentication, factory gateways, measured maintenance economics, cloud SLAs and safety-certified equipment control are outside the current deployment. Sensor aliases, inference provenance and coverage limitations remain explicit in the interface and documentation.
