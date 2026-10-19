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
