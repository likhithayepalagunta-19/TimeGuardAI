# TimeGuard AI: architecture, dataset strategy and research protocol

> **Final deployment synchronization — 20 October 2026:** The current system is the [Vercel portal](https://time-guard-ai.vercel.app/) backed by [Render Swagger API](https://timeguard-ai-api.onrender.com/docs) and [health](https://timeguard-ai-api.onrender.com/health). This dossier retains design alternatives and reference metadata; current implemented behavior, exact benchmarks and presentation claims are authoritative in the [README](../README.md), [benchmark report](BENCHMARK_RESULTS.md) and [Loom script](DEMO_SCRIPT.md). Median raw RMSE is 14.90 cycles, NASA S-Score 398.78, and nominal 90% intervals achieve 84% empirical coverage. The tau=.20 / 4:1 linear cost decision reduces late overestimates from 65% to 28%, with worse accuracy metrics. Fast Demo is verified-model recorded replay; TreeSHAP display aliases are not the physical metadata table below, and raw IDs remain preserved. CSV is an inference/decision audit for maintenance review. TWF/RNF recall is zero at threshold 0.5; HDF 97.5% and OSF 91.9% are individual uncalibrated preset scores.


## 1. Recommended concept

**TimeGuard AI: uncertainty-aware maintenance windows with cost-sensitive intervention.** Build a small, testable decision system around a strong CPU model. The differentiation is the connection between forecast, trust and action, demonstrated through controlled replay and compared against simpler baselines. Do not sell a new algorithm without evidence. Avoid a sprawling digital twin, LLM maintenance chatbot or transformer trained overnight.

### Industrial pain points and the corresponding design

| Pain point | Why a conventional forecast fails | TimeGuard design | What to measure |
|---|---|---|---|
| Operating changes and calibration drift | A sensor shift can resemble wear, making confidence misleading | Keep operating settings, fit normalizers on training data, add input-quality/shift indicators and review state | Coverage and error before/after controlled shift, review rate |
| Noisy multivariate telemetry | Single spikes trigger alarms, correlated channels confuse explanations | Causal EMA, robust bounds, trailing statistics and persistent alert policy | Noise ablation, false alert episodes |
| Missing, duplicate or stale packets | Silent interpolation can hide degraded observability | Per-asset sequence contract, reject duplicate/out-of-order packets in a real adapter, expose staleness, bounded hold-last prediction | Packet acceptance rate, stale duration, gated actions |
| Unequal error costs | Average squared error treats early and late errors equally | Conservative decision quantile and disclosed maintenance cost model | Late-estimate rate, NASA score, simulated total cost |
| Alert fatigue | Every tick becomes a fresh notification | Three consecutive crossings to warn, five safe ticks to clear, one event per state transition | Episodes per 1,000 cycles, time in warning |
| Poor interpretability | Operators cannot connect a score to observed behavior | Local TreeSHAP mapped to subsystem groups with feature value and trend | Attribution latency, stability and operator comprehension |
| Multiple degradation modes | A single label hides concurrent causes | Multilabel manufacturing model, domain-specific RUL | Per-mode support, recall, macro/micro F1, PR-AUC |
| Sparse failure outcomes | Validation leaks between almost identical windows | Split by asset, keep official endpoints untouched, report uncertainty of metrics | Group IDs, endpoint counts, bootstrap intervals |
| Maintenance censoring | Real assets often get repaired before failure | Explicitly defer survival modeling until censored operational data exists | Future time-dependent concordance and calibration |

An alert is a request for human maintenance review. RUL units remain cycles until a site validates a conversion to elapsed service time. Never advertise a safety-certified automatic shutdown.

## 2. Benchmark comparison and immediate acquisition

| Dataset | Data / supervision | Advantages | Constraints | Verdict |
|---|---|---|---|---|
| NASA C-MAPSS | Simulated multivariate engine trajectories with run-to-failure training and truncated test + RUL | Natural sequential replay, established RUL protocol, compact whitespace files | Simulated aerospace domain, FD001 has only one condition and HPC degradation, generic fault labels are unavailable per row | Primary benchmark: FD001 |
| UCI AI4I 2020 | 10,000 synthetic process snapshots, six usable base features and binary/mode labels | Tiny direct ZIP, understandable process variables, five failure modes | No trustworthy per-machine run-to-failure trajectory, rare RNF, deterministic label rules encourage high scores without broad generalization | Secondary diagnosis workbench |
| PRONOSTIA / FEMTO | Accelerated physical bearing life, intermittent high-rate vibration bursts and temperature | Actual rotating hardware, useful external domain stress test | Larger files, small number of independent bearing lives, feature extraction and acquisition need care | Stretch after core delivery |

**Best strategy:** FD001 for sequential RUL + AI4I for independent multilabel diagnosis. A common UI accepts domain-specific outputs. Never concatenate the datasets, invent engine identities in AI4I, or apply AI4I failure labels to C-MAPSS. FD003/004 include multiple degradation mechanisms but do not provide ready-made per-cycle failure-mode targets. Do not infer those targets without a defensible labeling source.

### Direct commands

The included loader uses standard-library HTTP and ZIP handling and records SHA256. No Kaggle account or scraping is needed for the core datasets.

```powershell
python scripts/download_data.py --dataset all
# or separate acquisition
curl.exe -L --fail 'https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip' -o ai4i.zip
curl.exe -L --fail 'https://data.nasa.gov/docs/legacy/CMAPSSData.zip' -o CMAPSSData.zip
```

NASA uses a redirect which can fail in restricted clients. Do not claim universal zero-delay availability. The catalog currently advertises that ZIP URL. The loader verifies required files after download. An HTML error saved as a ZIP must fail validation. Alternate official entry: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ .

FEMTO reference: https://members.femto-st.fr/patrick-nectoux/fr/pronostic-defaillances-roulement-pronostia . Distribution mirror linked from the NASA/PHM repository: https://data.phmsociety.org/nasa/ . Resolve the `FEMTO Bearing.zip` download from the repository page and inspect archive filenames, size and source before use. No verified one-click FEMTO binary URL is asserted here, and it is intentionally excluded from the six-hour critical path. Cite Nectoux et al., *PRONOSTIA: An Experimental Platform for Bearings Accelerated Life Test*, PHM 2012. Check distribution terms separately from code licensing.

### Exact C-MAPSS schema

Each text row has **26 whitespace-separated numeric columns** with possible trailing spaces: `unit`, `cycle`, `setting_1`, `setting_2`, `setting_3`, `s1` ... `s21`. Use `sep=r'\s+'`, no header. Engine IDs restart between subsets and between train and test, so include split/subset in any global identity. RUL files contain one number per test engine in ascending unit order: remaining life after its final observed cycle.

Common sensor metadata (verify against the distributed readme before attaching physical units):

| Field | Typical symbol / interpretation | Typical unit |
|---|---|---|
| s1 | T2, fan inlet temperature | degrees R |
| s2 | T24, LPC outlet temperature | degrees R |
| s3 | T30, HPC outlet temperature | degrees R |
| s4 | T50, LPT outlet temperature | degrees R |
| s5 | P2, fan inlet pressure | psia |
| s6 | P15, bypass-duct pressure | psia |
| s7 | P30, HPC outlet pressure | psia |
| s8 | Nf, physical fan speed | rpm |
| s9 | Nc, physical core speed | rpm |
| s10 | epr, engine pressure ratio | ratio |
| s11 | Ps30, HPC outlet static pressure | psia |
| s12 | phi, fuel-flow to Ps30 ratio | flow/pressure ratio |
| s13 | NRf, corrected fan speed | rpm |
| s14 | NRc, corrected core speed | rpm |
| s15 | BPR, bypass ratio | ratio |
| s16 | farB, burner fuel-air ratio | ratio |
| s17 | htBleed, bleed enthalpy | dataset-defined |
| s18 | Nf_dmd, demanded fan speed | rpm |
| s19 | PCNfR_dmd, demanded corrected fan speed | percent |
| s20 | W31, HPT coolant bleed | lbm/s |
| s21 | W32, LPT coolant bleed | lbm/s |

FD001 has 100 training and 100 test engines, one condition/one mode. FD002: 260/259, six conditions/one mode. FD003: 100/100, one condition/two modes. FD004: 248/249, six conditions/two modes. Verify counts from downloaded files. FD001 candidate informative sensors: `s2,s3,s4,s7,s8,s9,s11,s12,s13,s14,s15,s17,s20,s21`. Do not blindly drop the other sensors on all subsets. Settings are retained as context features in the supplied implementation, though nearly constant on FD001.

For full test trajectory labels used for offline plots only, `rul_t = official_endpoint_rul + last_observed_cycle - cycle_t`. Do not use `last_observed_cycle` as an online feature. Training RUL uses full training lifetimes solely to construct targets.

### Exact AI4I CSV schema

`UDI`, `Product ID`, `Type`, `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Machine failure`, `TWF`, `HDF`, `PWF`, `OSF`, `RNF`.

Base inputs are Type (L/M/H) and the five measured process variables. Drop UDI and Product ID. Binary target is Machine failure. Mode targets are the five last columns, with independent Bernoulli classifiers or a multioutput forest. The mode labels must never enter the binary failure features. Temperature gap is process minus air temperature, mechanical power is `torque * rpm * 2*pi/60`, and wear-load product is `wear*torque`.

AI4I's generated mechanisms include wear, heat dissipation, power and overstrain thresholds. A model may recover those rules. This supports a transparent demo, not an industrial generalization claim. Label inconsistencies and overlapping modes should be audited instead of forcing an exclusive class or silently rebuilding the binary target from mode OR. RNF has very few positives and occurs independently of physical inputs. Report its support and inability to achieve reliable diagnosis. Stratify binary splits, disclose random row split limitations from generated dependencies, and add contiguous-block sensitivity analysis if time permits. Do not compute rolling machine RUL from row order.

### PRONOSTIA schema and preprocessing

The challenge has six complete training bearing lives and eleven truncated test lives across three operating conditions. Accelerometer CSV bursts typically contain `hour, minute, second, microsecond, horizontal_acceleration, vertical_acceleration`. Each burst has 2,560 points sampled at 25.6 kHz, recording 0.1 seconds every 10 seconds. Temperature files have their own sampling and column contract, typically time fields plus temperature. Validate delimiters and headers against the particular distribution, not a universal pandas assumption. Extract burst-level RMS, kurtosis, crest factor and spectral band energy. Split by bearing, retain condition metadata and label remaining life in **seconds or bursts**, not turbofan cycles. A small life-level sample makes calibration uncertain.

## 3. Causal feature pipeline

1. Validate finite numeric fields, unique `(unit,cycle)`, strictly increasing cycles per unit and contiguous history. Sort offline reference data once. A production packet adapter must reject/reorder against a bounded buffer rather than hiding late arrival.
2. Fit median/IQR or standard scaling on fit engines only for a neural alternative. Tree models do not need feature scaling. Keep data-quality flags and missingness independent of the regressor.
3. Causal EMA: `z_t = 0.25*x_t + 0.75*z_(t-1)`, initialize at first measurement. Reset state between assets and never pool EMA histories across engines. Avoid excessive smoothing that delays warning.
4. Savitzky–Golay is an optional comparison. The usual centered implementation uses future values. If used online, apply a trailing window and evaluate the polynomial only at its final sample. The delivered default is EMA.
5. For a 30-cycle trailing window calculate last, mean, sample standard deviation, min, max and least-squares slope over time. Add skewness/kurtosis only after checking variance and stability. More moments are not automatically more useful. The supplied minimal pipeline implements the six robust features.
6. Withhold prediction during warmup. Do not backfill the first 29 cycles with future values. Report eligible engine count if a test engine is shorter than the selected window.
7. For a temporal neural model use a causal 30×14 tensor, a two-block dilated TCN with 32 channels and kernel 3, dilation 1/2, global last-state pooling, and quantile heads. Train under 15 minutes with early stopping on held-out engines. This is a stretch experiment, without an invented implementation or result claim.

## 4. Objectives and decision policy

Let `d = predicted_RUL - true_RUL`. Positive d means predicting failure too late.

**Asymmetric squared alternative:** `L = 4*max(d,0)^2 + max(-d,0)^2`. An XGBoost custom objective can use gradient `2*w*d` and Hessian `2*w` with `w=4` for positive d else 1. This minimizes an expectile objective, not a quantile. Do not attach quantile coverage to it.

**Delivered cost decision:** `C = c_late*max(d,0) + c_early*max(-d,0)`. Set illustrative `c_late=4`, `c_early=1`, giving quantile `tau=c_early/(c_late+c_early)=0.20`. Pinball loss is `rho_tau(u)=max(tau*u,(tau-1)*u)` for `u=y-q`. Thus tau below 0.5 favors conservative lower RUL. Keep a separate median for accuracy and outer quantiles for uncertainty. A low RUL quantile is not a calibrated probability of failure in the next N cycles.

**NASA score:** sum `exp(-d/13)-1` when `d<0`, else `exp(d/10)-1`. Lower is better. Report official raw-RUL score and state when RMSE uses a cap. Never compare clipped and raw protocols as identical.

**Policy:** after three valid consecutive cycles with calibrated lower RUL bound ≤ maintenance horizon, request maintenance review. Clear only after five valid consecutive cycles with lower bound > horizon+10. Bad/missing telemetry triggers REVIEW, freezes actionable escalation and retains freshness context. Never reset a warning by treating a missing packet as recovery. These are prototype rules that require validation. Tune horizon, persistence and threshold on validation, lock before test evaluation. Acknowledge acknowledges the event, not the physical fault.

**Cost evaluation:** compare run-to-failure, fixed threshold, median-only and conservative-bound policies on identical held-out trajectories. Publish planned maintenance cost, failure cost, downtime cost per hour, service duration and premature replacement penalty. Count one intervention per episode, prevent impossible repeated maintenance on the same unrepaired asset, and show sensitivity to costs. RUL regression error alone cannot prove annual savings.

## 5. Conformal intervals without overstated guarantees

Fit quantiles q05 and q95 on fitting engines. On n independent calibration endpoints, compute `s_i=max(q05_i-y_i, y_i-q95_i, 0)`. Use `k=ceil((n+1)*(1-alpha))` with alpha=0.10, and the k-th smallest calibration score, not an interpolated percentile. If k>n, the valid conservative correction is infinity. Sort quantile endpoints first to avoid crossing. Return `[max(0,q05-qhat), q95+qhat]`; do not truncate the upper endpoint to 125 when evaluating raw RUL because it can destroy coverage.

The implementation calibrates against **raw** endpoint RUL even though the quantile models train on capped labels. This can widen high-RUL intervals and needs explicit reporting. A cleaner future option is a separate uncapped model or a clearly stated capped-target coverage claim.

Windows from the same engine are dependent. Do not treat 10,000 overlapping windows as 10,000 calibration units. Use one preregistered truncated endpoint per engine for a marginal endpoint claim. Their truncation distribution may differ from NASA test truncation, so report the distribution and actual test coverage. For full-trajectory coverage, a different group-level maximum score can give conservative simultaneous coverage under exchangeability of complete trajectories, but it is not what the current endpoint calibrator implements. Real drift invalidates ordinary exchangeability. Drift-aware adaptive conformal is future work and still requires assumptions and delayed labels.

Report interval coverage together with mean/median width, test-engine sample size, a binomial interval as a rough endpoint uncertainty summary, and coverage slices by short/long RUL. Never display “90% certain this machine fails here” or claim conditional guarantees for a specific machine.

## 6. Attribution and failure diagnosis

Use TreeSHAP on the actual fitted RUL estimator with fixed model version and feature names. Report sign: a negative contribution to RUL reduces predicted life. Aggregate mean/slope contributions back to sensor, then optionally into thermal (`s2,s3,s4`), pressure (`s7,s11`), rotational (`s8,s9,s13,s14`) and other/flow (`s12,s15,s17,s20,s21`) groups. Keep the ungrouped view accessible. These are explanatory categories, not proven root causes. Correlated engineered windows can split importance across features. Cache or throttle explanation to about once per second and measure explanation latency separately from prediction.

Do not call a neural attention map an equivalent proof of sensor causality. A TCN with attention still needs perturbation tests, stable attribution and comparison to baseline explanations. In the Stage 1 portal, contribution bars are explicitly heuristic scenario weights, not TreeSHAP.

## 7. Evaluation and ablation matrix

| Experiment | Purpose | Minimum evidence |
|---|---|---|
| Constant median and squared-error boosted baseline | Establish meaningful performance floor | Endpoint RMSE and NASA score |
| Median + causal windows | Measure temporal feature benefit | Same engine split and runtime budget |
| Conservative tau=0.20 estimate | Isolate asymmetric decision effect | Late rate, early error and cost sensitivity |
| Raw quantiles vs conformal correction | Isolate calibration benefit | Coverage plus width |
| Single-tick vs persistent policy | Isolate alert-fatigue benefit | Episode counts and delay |
| Clean vs noise/offset/missing replay | Reveal robustness limits | Error, review duration and coverage shift |
| AI4I physical vs engineered features | Identify rule reconstruction benefit | Same held-out rows, per-mode results |

Use held-out engine bootstrap intervals for RUL comparisons if time permits. Report feature+inference latency separately from model-only and SHAP latency. Do not average row-level scores to inflate effective sample size. Record CPU, Python/library versions, random seeds, git commit, data hashes and splits. A short deterministic replay and complete evidence table will score better than an unvalidated complex model.

## 8. Feasibility, positioning and risks

The six-hour critical path is FD001, causal windows, CPU quantiles, calibration, simple real replay and transparent evaluation. AI4I classification is inexpensive, but complete it only after preserving core RUL quality. Reserve 90 minutes of the stated 7.5-hour window for bug fixes, screenshots, submission and rehearsal. Do not let PRONOSTIA download, a transformer, cloud hosting or an LLM consume the core window.

Stage 1 must communicate the pain, differentiated method, data feasibility, honest targets and demo. Stage 2 must show measured outputs, reproduce them from commands and explain failure cases. No method guarantees a win. Check the actual organizer rules before using prepared code during a timed round or submitting AI-assisted content.

## 9. Sources

- NASA catalog: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- NASA repository: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- Saxena et al., PHM 2008: https://ntrs.nasa.gov/citations/20090029214
- UCI AI4I, DOI 10.24432/C5HS5C: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- PRONOSTIA creator reference: https://members.femto-st.fr/patrick-nectoux/fr/pronostic-defaillances-roulement-pronostia
- CQR research: https://arxiv.org/abs/1905.03222
- TreeSHAP API: https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html
- Quantile estimator API: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

Sources checked on 19 October 2026. Download success is separately recorded by the loader, not implied by a catalog listing.
