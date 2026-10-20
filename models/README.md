# Verified deployment models

`rul.joblib` and `modes.joblib` are tracked deployment artifacts, explicitly authorized by the team. `manifest.json` records their SHA256 hashes, compatible package versions, model version and the committed demo/evaluation data hashes.

The API and evaluator verify bytes before loading. They never download or train at boot, and never substitute synthetic predictions for missing models. Only load trusted repository artifacts. Joblib is not safe for user-supplied model files.

`rul.joblib`: four quantile regressors, feature order, 30-cycle contract, conformal correction and original engine split IDs.

`modes.joblib`: binary and five separate failure-mode forests, feature order and original holdout indices. Runtime sets forest n_jobs=1 for predictable single-process CPU deployment.

Retraining is optional: run the documented train commands using raw data, then `python scripts/package_demo.py`, re-evaluate on a declared independent protocol and review all changed results/manifests. Do not silently replace the published experiment.
