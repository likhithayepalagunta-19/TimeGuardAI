# Deployment upgrade verification — 20 October 2026

- Real FastAPI endpoint tests cover ready/unavailable state, CORS, typed results, actual TreeSHAP additivity, custom prefix parity, isolated policy reconstruction, missing-packet hold, causal hold-last input imputation, offset review and corrupted model rejection.
- `python evaluate.py` reproduces the original 100-engine FD001 and 2,000-row AI4I holdout metrics without downloads or training. Machine-readable outputs are in reports/verification*.
- Nominal 90% interval coverage remains 84%. TWF and RNF recall remain zero at threshold 0.5. Historical RUL baseline outperforms the quantile median on RMSE and NASA score.
- Browser verification covers full live RUL replay, custom live manufacturing inputs, unavailable-API fallback, stale-packet holds and CSV export. Both narrow/mobile and 1280px desktop layouts were inspected.
- Offline fallback uses real recorded model predictions and TreeSHAP. It withholds unsupported custom manufacturing probabilities and labels unscored sensor offsets.
- `models/manifest.json` verifies exact model, demo data, original metric-report and offline-prediction bytes. Git attributes preserve those bytes across Windows/Linux.
- Models, demo slices, inference/evaluation source, reports and deployment configs are intended to be tracked. Raw downloads and installed environments remain ignored.
- Render and Vercel configuration is prepared. No claim of an actual hosted deployment, cold-start duration, production load test or Render end-to-end latency is made.

## Earlier screening-stage verification

# Verification record

19 October 2026.

- Four core tests pass: prefix invariance of causal features, conformal finite-sample order statistic, asymmetric NASA penalty, and alert persistence/recovery after missing telemetry.
- All Python source files compile and the browser JavaScript passes its syntax check.
- Browser verification covers step, accelerated replay through completion, reset, sensor-offset review, missing-packet forecast hold and successful CSV download.
- Stage 1 portal screenshot is in `docs/screenshots/portal.png`.
- The 12-slide PowerPoint passes package integrity, geometry/font checks and first-party reimport. Every rendered slide was visually inspected. This is not a claim of inspection in Microsoft PowerPoint.
- PDF is an image-based convenience export of the inspected slides. Use the PPTX for editable content and native flowchart objects.
- Official C-MAPSS and AI4I downloads validated required filenames and archive structure. SHA256 provenance is retained in `reports/data_provenance.json`.
- Actual benchmark outputs are saved in `reports/`. They are specific to the recorded splits and environment. Streamlit source compiles, but the Streamlit UI itself has not been interactively inspected in this run.
- RUL evaluation completed on 100 engines. Official-test coverage is 84% for nominal 90%, an unmet target. AI4I evaluation completed on 2,000 holdout rows. Keep zero TWF/RNF recall visible.
- TreeSHAP export has not been executed. Full maintenance-cost simulation and end-to-end latency remain unmeasured.

Canonical destination verification:
- Installed project dependencies in C:\Users\likhi\OneDrive\Documents\TimeGuard\.venv.
- Four core tests pass using that environment.
- Headless Streamlit checks pass: load actual RUL weights, advance one replay cycle, and run manufacturing diagnosis. This does not claim interactive browser inspection of Streamlit.
- Local and GitHub main are synchronized. Raw data, model binaries and the environment remain local and ignored.

Linux CI: GitHub Actions run 39377261783 passed the real endpoint/core tests, original benchmark evaluation, and JavaScript syntax check on commit 6a884ec.
