"""One command, no downloads or training: recompute the committed holdout results."""
from pathlib import Path
import json
import platform
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from timeguard.features import ai4i_features, make_features
from timeguard.metrics import rmse, nasa_score
from timeguard.runtime import ModelRuntime


def evaluate(root=ROOT):
    runtime = ModelRuntime(root)
    endpoints = pd.read_csv(root / 'data/demo_slice/cmapss_endpoints.csv')
    forecasts = runtime.forecast(endpoints)
    truth = endpoints.raw_rul.to_numpy()
    metrics = {}
    for key in ['point', 'risk']:
        prediction = forecasts[key]
        metrics[key] = {'rmse_raw': rmse(truth, prediction),
                        'rmse_clipped': rmse(np.minimum(truth, 125), prediction),
                        'nasa_score_raw': nasa_score(truth, prediction),
                        'late_rate': float(np.mean(prediction > truth))}
    metrics['interval'] = {'nominal_coverage': .9,
        'coverage': float(np.mean((forecasts['lower'] <= truth) & (truth <= forecasts['upper']))),
        'mean_width': float(np.mean(forecasts['upper'] - forecasts['lower']))}
    # Check the frozen endpoint features against raw-prefix preprocessing on every demo engine.
    for unit, raw in runtime.raw.groupby('unit'):
        actual = make_features(raw).tail(1)[runtime.rul['features']].to_numpy()
        expected = endpoints[endpoints.unit == unit][runtime.rul['features']].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    rul_rows = endpoints[['unit', 'cycle', 'raw_rul']].copy()
    for key, array in forecasts.items():
        rul_rows[key] = array
    holdout = pd.read_csv(root / 'data/demo_slice/ai4i_holdout.csv')
    x = ai4i_features(holdout)[runtime.modes['features']]
    mode_rows = holdout[['UDI', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']].copy()
    labels = {}
    for label, model in [('binary', runtime.modes['binary']), *runtime.modes['modes'].items()]:
        target = 'Machine failure' if label == 'binary' else label
        y = holdout[target].to_numpy()
        p = model.predict_proba(x)[:, list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(y))
        precision, recall, f1, _ = precision_recall_fscore_support(y, p >= .5, average='binary', zero_division=0)
        labels[label] = {'precision': float(precision), 'recall': float(recall), 'f1': float(f1),
                         'pr_auc': float(average_precision_score(y, p)) if np.any(y) else None,
                         'positive_support': int(y.sum())}
        mode_rows[label + '_probability'] = p
    checks = []
    for name, old in [('point', 'median'), ('risk', 'decision_tau_020')]:
        for metric, value in metrics[name].items():
            checks.append(bool(np.isclose(value, runtime.summary[old][metric], rtol=1e-6, atol=1e-6)))
    checks.append(bool(np.isclose(metrics['interval']['coverage'], runtime.summary['coverage_90_raw'])))
    checks.append(bool(np.isclose(metrics['interval']['mean_width'], runtime.summary['mean_width'])))
    for label, values in labels.items():
        for metric, value in values.items():
            old = runtime.mode_summary['labels'][label][metric]
            checks.append(value == old if value is None or old is None else bool(np.isclose(value, old, rtol=1e-6, atol=1e-6)))
    result = {'verified': all(checks), 'model_version': runtime.version,
        'protocol': 'Frozen official FD001 final endpoints (100 engines), original AI4I holdout (2000 rows). No training or test-driven recalibration.',
        'rul': metrics, 'ai4i': labels, 'counts': {'rul_engines': len(endpoints), 'ai4i_rows': len(holdout)},
        'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}}
    reports = root / 'reports'
    reports.mkdir(exist_ok=True)
    (reports / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8', newline='\n')
    rul_rows.to_csv(reports / 'verification_rul_predictions.csv', index=False, lineterminator='\n')
    mode_rows.to_csv(reports / 'verification_ai4i_predictions.csv', index=False, lineterminator='\n')
    table = [{'dataset': 'FD001', 'target': key, **values} for key, values in metrics.items()]
    table += [{'dataset': 'AI4I', 'target': key, **values} for key, values in labels.items()]
    pd.DataFrame(table).to_csv(reports / 'verification_metrics.csv', index=False, lineterminator='\n')
    print('\nTIMEGUARD | VERIFIED HOLDOUT EVALUATION\n')
    print(f"{'RUL model':<20} {'RMSE raw':>11} {'NASA score':>13} {'Late rate':>11}")
    for label, key in [('Median', 'point'), ('Risk tau=.20', 'risk')]:
        m = metrics[key]
        print(f"{label:<20} {m['rmse_raw']:>11.4f} {m['nasa_score_raw']:>13.4f} {m['late_rate']:>10.1%}")
    print(f"\nNominal 90% coverage: {metrics['interval']['coverage']:.1%}; mean width: {metrics['interval']['mean_width']:.2f} cycles")
    print('Coverage target missed. Conservative RUL trades accuracy for fewer late estimates.\n')
    print(f"{'AI4I target':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'PR-AUC':>10} {'Support':>9}")
    for label, m in labels.items():
        print(f"{label:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {(m['pr_auc'] or 0):>10.4f} {m['positive_support']:>9}")
    print('\nOriginal benchmark parity:', 'PASS' if result['verified'] else 'FAIL')
    print('JSON + per-row predictions + metric CSVs saved to reports/.')
    return result


if __name__ == '__main__':
    try:
        sys.exit(0 if evaluate()['verified'] else 1)
    except Exception as exc:
        print(f'Evaluation failed: {exc}', file=sys.stderr)
        sys.exit(1)
