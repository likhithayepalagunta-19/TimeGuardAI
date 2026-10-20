"""Explicit maintainer operation: export original holdouts and real-model offline replay."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import joblib
import numpy as np
import pandas as pd
from timeguard.features import read_cmapss, make_features


def package():
    data = ROOT / 'data/demo_slice'
    data.mkdir(parents=True, exist_ok=True)
    rul = joblib.load(ROOT / 'models/rul.joblib')
    modes = joblib.load(ROOT / 'models/modes.joblib')
    raw = read_cmapss(ROOT / 'data/raw/cmapss/test_FD001.txt')
    targets = np.loadtxt(ROOT / 'data/raw/cmapss/RUL_FD001.txt')
    endpoints = make_features(raw).groupby('unit', sort=True).tail(1).sort_values('unit').copy()
    endpoints['raw_rul'] = endpoints.unit.map(dict(zip(sorted(raw.unit.unique()), targets)))
    endpoints.to_csv(data / 'cmapss_endpoints.csv', index=False, float_format='%.17g')
    replay = raw[raw.unit.isin([1, 20, 81])]
    replay.to_csv(data / 'cmapss_replay.csv', index=False, float_format='%.17g')
    all_ai4i = pd.read_csv(ROOT / 'data/raw/ai4i/ai4i2020.csv')
    holdout = all_ai4i.iloc[modes['test_indices']].copy()
    holdout.to_csv(data / 'ai4i_holdout.csv', index=False)
    selected = []
    for name in ['HDF', 'PWF', 'OSF', 'TWF', 'RNF']:
        selected.extend([(int(idx), name+' benchmark example') for idx in holdout.index[holdout[name] == 1][:2]])
    selected.extend([(int(idx), 'Normal benchmark example') for idx in holdout.index[holdout['Machine failure'] == 0][:5]])
    presets, seen = [], set()
    for idx, name in selected:
        if idx in seen:
            continue
        seen.add(idx)
        row = all_ai4i.loc[idx]
        presets.append({'id': int(row.UDI), 'name': f'{name} · UDI {int(row.UDI)}',
                        'telemetry': {'type': row.Type, 'air_temperature': float(row['Air temperature [K]']),
                         'process_temperature': float(row['Process temperature [K]']),
                         'rotational_speed': float(row['Rotational speed [rpm]']), 'torque': float(row['Torque [Nm]']),
                         'tool_wear': float(row['Tool wear [min]'])}})
    (data / 'ai4i_presets.json').write_text(json.dumps(presets, indent=2), encoding='utf-8')
    packages = ['numpy', 'pandas', 'scikit-learn', 'scipy', 'joblib', 'shap']
    manifest = {'schema_version': 1, 'model_version': 'fd001-ai4i-'+hashlib.sha256((ROOT/'models/rul.joblib').read_bytes()).hexdigest()[:12],
                'environment': {p: importlib.metadata.version(p) for p in packages},
                'counts': {'rul_endpoints': len(endpoints), 'ai4i_holdout': len(holdout), 'replay_engines': 3}, 'files': {}}
    files = ['models/rul.joblib', 'models/modes.joblib',
             'data/demo_slice/cmapss_endpoints.csv', 'data/demo_slice/cmapss_replay.csv',
             'data/demo_slice/ai4i_holdout.csv', 'data/demo_slice/ai4i_presets.json',
             'reports/rul.json', 'reports/modes.json']
    def record(name):
        blob = (ROOT / name).read_bytes()
        if not name.endswith('.joblib'):
            blob = blob.replace(b'\r\n', b'\n')
            (ROOT / name).write_bytes(blob)
        manifest['files'][name] = {'sha256': hashlib.sha256(blob).hexdigest(), 'size_bytes': len(blob)}
    for name in files:
        record(name)
    (ROOT/'models/manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    from timeguard.runtime import ModelRuntime
    runtime = ModelRuntime(ROOT)
    offline = {'model_version': runtime.version, 'provenance': 'recorded_real_model',
               'nominal_coverage': .9, 'empirical_coverage': .84, 'engines': [], 'ai4i': []}
    # Compute TreeSHAP in one batch per engine, then aggregate consistently to sensors.
    for engine, (features, predictions) in runtime.clean.items():
        values = runtime.explainer(features[rul['features']])
        rows = []
        for i, cycle in enumerate(features.cycle):
            grouped = {}
            for feature, value in zip(rul['features'], values.values[i]):
                sensor = feature.rsplit('_', 1)[0] if not feature.startswith('setting_') else feature
                grouped[sensor] = grouped.get(sensor, 0.) + float(value)
            top = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[:8]
            rows.append({'cycle': int(cycle), 'point_rul': float(predictions['point'][i]),
                         'risk_aware_rul': float(predictions['risk'][i]),
                         'interval': {'lower': float(predictions['lower'][i]), 'upper': float(predictions['upper'][i])},
                         'attribution': {'method': 'TreeSHAP', 'top': [{'sensor': k, 'contribution': v} for k, v in top]}})
        offline['engines'].append({'id': engine, 'min_cycle': 30, 'max_cycle': int(features.cycle.max()), 'rows': rows})
    for preset in presets:
        offline['ai4i'].append({**preset, 'prediction': runtime.diagnose(preset['telemetry'])})
    (ROOT/'portal/offline-data.js').write_text('window.TIMEGUARD_OFFLINE = '+json.dumps(offline, separators=(',', ':'), allow_nan=False)+';\n', encoding='utf-8')
    record('portal/offline-data.js')
    (ROOT/'models/manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Packaged', manifest['counts'], 'model version', runtime.version)
    print('Offline predictions and TreeSHAP are actual recorded outputs, not synthetic estimates.')


if __name__ == '__main__':
    package()
