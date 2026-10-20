"""Verified, read-only model runtime shared by the API and evaluator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import RLock

import joblib
import numpy as np
import pandas as pd

from .conformal import intervals
from .features import COLUMNS, ai4i_features, make_features
from .policy import MaintenancePolicy


def verify_manifest(root: Path) -> dict:
    manifest = json.loads((root / 'models/manifest.json').read_text(encoding='utf-8'))
    for name, record in manifest['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('Invalid artifact path')
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Artifact verification failed: {name}')
    return manifest


class ModelRuntime:
    """No training, download, global alert state, or synthetic backend predictions."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.manifest = verify_manifest(self.root)
        import sklearn
        if sklearn.__version__ != self.manifest['environment']['scikit-learn']:
            raise ValueError('Installed scikit-learn does not match the verified model environment')
        self.version = self.manifest['model_version']
        self.rul = joblib.load(self.root / 'models/rul.joblib')
        self.modes = joblib.load(self.root / 'models/modes.joblib')
        for clf in [self.modes['binary'], *self.modes['modes'].values()]:
            clf.n_jobs = 1
        self.raw = pd.read_csv(self.root / 'data/demo_slice/cmapss_replay.csv')
        self.snapshots = json.loads((self.root / 'data/demo_slice/ai4i_presets.json').read_text())
        self.summary = json.loads((self.root / 'reports/rul.json').read_text())
        self.mode_summary = json.loads((self.root / 'reports/modes.json').read_text())
        self.lock = RLock()
        # TreeSHAP explains the actual fitted median model. Failure blocks readiness.
        import shap
        self.explainer = shap.TreeExplainer(self.rul['models'][.5])
        self.clean = {}
        for unit, frame in self.raw.groupby('unit', sort=True):
            f = make_features(frame, self.rul['window'])
            self.clean[int(unit)] = (f, self.forecast(f))
        probe = next(iter(self.clean.values()))[0].tail(1)
        self.attributions(probe)

    def forecast(self, features: pd.DataFrame) -> dict:
        x = features[self.rul['features']]
        predictions = {q: model.predict(x) for q, model in self.rul['models'].items()}
        lower, upper = intervals(predictions[.05], predictions[.95], self.rul['qhat'])
        return {'point': predictions[.5], 'risk': predictions[.2], 'lower': lower, 'upper': upper}

    def attributions(self, features: pd.DataFrame) -> dict:
        x = features.tail(1)[self.rul['features']]
        with self.lock:
            explanation = self.explainer(x)
        values = explanation.values[0]
        grouped = {}
        for name, value in zip(x.columns, values):
            sensor = name.rsplit('_', 1)[0] if not name.startswith('setting_') else name
            grouped[sensor] = grouped.get(sensor, 0.) + float(value)
        ranked = sorted(grouped.items(), key=lambda p: abs(p[1]), reverse=True)
        return {
            'method': 'TreeSHAP', 'units': 'RUL cycles',
            'baseline_rul': float(np.asarray(explanation.base_values).reshape(-1)[0]),
            'all_contributions_sum': float(values.sum()),
            'top': [{'sensor': name, 'contribution': value} for name, value in ranked[:8]],
            'note': 'Negative contributions reduce median RUL. Model attribution is not physical causality.'
        }

    def engine_list(self) -> dict:
        return {
            'model_version': self.version,
            'engines': [{'id': int(k), 'min_cycle': self.rul['window'],
                         'max_cycle': int(g.cycle.max()), 'name': f'FD001 engine {int(k):03d}'}
                        for k, g in self.raw.groupby('unit', sort=True)],
            'manufacturing_samples': self.snapshots,
            'evaluation_counts': {'rul_engines': 100, 'ai4i_holdout_rows': 2000}
        }

    def predict_rul(self, engine_id=1, cycle=30, observations=None, offsets=None,
                    packet_loss=False, horizon=45, explain=True) -> dict:
        offsets = offsets or {}
        if observations is None:
            if engine_id not in self.clean:
                raise LookupError('Engine is not in the committed demo slice')
            raw = self.raw[self.raw.unit == engine_id]
            if cycle < self.rul['window'] or cycle > int(raw.cycle.max()):
                raise ValueError('Cycle is outside the observed demo prefix')
            prefix = raw[raw.cycle <= cycle].copy()
        else:
            prefix = pd.DataFrame(observations, columns=COLUMNS)
            if not np.isfinite(prefix.to_numpy(dtype=float)).all():
                raise ValueError('Every observation must contain finite numeric values')
            if prefix.unit.nunique() != 1 or not np.all(np.diff(prefix.cycle) == 1):
                raise ValueError('Provide one engine with ordered contiguous cycles')
            if (prefix.unit.iloc[0] < 1 or prefix.unit.iloc[0] != int(prefix.unit.iloc[0]) or prefix.cycle.iloc[0] < 1):
                raise ValueError('Unit and cycle must be positive integers')
            if not np.all(prefix.cycle == prefix.cycle.astype(int)):
                raise ValueError('Cycles must be integers')
            cycle, engine_id = int(prefix.cycle.iloc[-1]), int(prefix.unit.iloc[-1])
        if offsets:
            for sensor, value in offsets.items():
                if sensor not in COLUMNS[5:]:
                    raise ValueError('Offsets must name sensor s1 through s21')
                prefix.loc[prefix.index[-5:], sensor] += value
        if packet_loss:
            lost = prefix.cycle.astype(int) % 3 == 0
            prefix.loc[lost, COLUMNS[2:]] = np.nan
            prefix[COLUMNS[2:]] = prefix[COLUMNS[2:]].ffill()
            if prefix[COLUMNS[2:]].isna().any().any():
                raise ValueError('Packet-loss replay requires an earlier valid observation for causal hold-last imputation')
        if observations is None and not offsets and not packet_loss:
            all_features, all_forecasts = self.clean[engine_id]
            mask = (all_features.cycle <= cycle).to_numpy()
            features = all_features.loc[mask]
            forecasts = {k: v[mask] for k, v in all_forecasts.items()}
        else:
            features = make_features(prefix, self.rul['window'])
            forecasts = self.forecast(features)
        if features.empty:
            raise ValueError('At least 30 observed cycles are required')
        policy = MaintenancePolicy(horizon=horizon)
        last_valid = None
        for i, step in enumerate(features.cycle):
            missing = packet_loss and int(step) % 3 == 0
            if not missing:
                last_valid = i
            policy.update(float(forecasts['lower'][i]), valid=not missing and not bool(offsets))
        stale = packet_loss and cycle % 3 == 0
        if last_valid is None:
            return {'engine_id': engine_id, 'cycle': cycle, 'observed_cycle': None,
                    'point_rul': None, 'risk_aware_rul': None, 'interval': None,
                    'alert_state': 'REVIEW', 'quality': 'STALE', 'attribution': None,
                    'model_version': self.version, 'provenance': 'live_model',
                    'note': 'No valid packet after warmup. Prediction withheld.'}
        i = last_valid
        attribution = self.attributions(features.iloc[[i]]) if explain else None
        return {
            'engine_id': engine_id, 'cycle': cycle, 'observed_cycle': int(features.cycle.iloc[i]),
            'point_rul': float(forecasts['point'][i]), 'risk_aware_rul': float(forecasts['risk'][i]),
            'interval': {'lower': float(forecasts['lower'][i]), 'upper': float(forecasts['upper'][i]),
                         'nominal_coverage': .9, 'benchmark_empirical_coverage': .84,
                         'clean_benchmark_request': not packet_loss and not bool(offsets)},
            'alert_state': policy.state, 'quality': 'STALE' if stale else 'SHIFT_REVIEW' if offsets else 'CURRENT',
            'attribution': attribution, 'model_version': self.version, 'provenance': 'live_model',
            'note': 'Nominal coverage is not guaranteed for a particular engine or under injected shift. Risk-aware RUL is a cost quantile, not the NASA score.'
        }

    def diagnose(self, telemetry: dict) -> dict:
        frame = pd.DataFrame([{'Type': telemetry['type'],
            'Air temperature [K]': telemetry['air_temperature'],
            'Process temperature [K]': telemetry['process_temperature'],
            'Rotational speed [rpm]': telemetry['rotational_speed'],
            'Torque [Nm]': telemetry['torque'], 'Tool wear [min]': telemetry['tool_wear']}])
        x = ai4i_features(frame)[self.modes['features']]
        def probability(model):
            return float(model.predict_proba(x)[0, list(model.classes_).index(1)]) if 1 in model.classes_ else 0.
        modes = {name: probability(model) for name, model in self.modes['modes'].items()}
        return {'failure_probability': probability(self.modes['binary']), 'modes': modes,
                'predicted_modes': [name for name, p in modes.items() if p >= .5],
                'threshold': .5, 'probabilities_calibrated': False,
                'model_version': self.version, 'provenance': 'live_model',
                'note': 'Synthetic AI4I benchmark only. Modes overlap. TWF/RNF holdout recall is zero at threshold 0.5. Low scores are not proof of safe operation.'}
