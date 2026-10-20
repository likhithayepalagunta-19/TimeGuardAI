import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
from api.main import create_app
from timeguard.runtime import verify_manifest


@pytest.fixture(scope='module')
def client():
    with TestClient(create_app()) as value:
        assert value.get('/health').status_code == 200
        yield value


def test_health_and_cors(client):
    health = client.get('/health').json()
    assert health['ready'] and health['benchmark']['empirical_coverage'] == .84
    assert 'TreeSHAP' in health['loaded_artifacts']
    cors = client.options('/api/predict/rul', headers={'Origin':'https://team.vercel.app', 'Access-Control-Request-Method':'POST', 'Access-Control-Request-Headers':'content-type'})
    assert cors.headers['access-control-allow-origin'] == '*'
    assert 'access-control-allow-credentials' not in cors.headers


def test_engines_are_real_committed_prefixes(client):
    value = client.get('/api/engines').json()
    assert [engine['id'] for engine in value['engines']] == [1, 20, 81]
    assert len(value['manufacturing_samples']) >= 5


def test_rul_and_actual_shap_additivity(client):
    response = client.post('/api/predict/rul', json={'engine_id':20, 'cycle':60})
    assert response.status_code == 200
    value = response.json()
    assert value['provenance'] == 'live_model'
    assert value['interval']['lower'] <= value['interval']['upper']
    assert value['attribution']['method'] == 'TreeSHAP'
    explanation = value['attribution']
    assert np.isclose(explanation['baseline_rul'] + explanation['all_contributions_sum'], value['point_rul'])


def test_packet_loss_holds_previous_valid_prediction(client):
    prior = client.post('/api/predict/rul', json={'engine_id':20, 'cycle':32, 'packet_loss':True}).json()
    missing = client.post('/api/predict/rul', json={'engine_id':20, 'cycle':33, 'packet_loss':True}).json()
    assert missing['quality'] == 'STALE' and missing['alert_state'] == 'REVIEW'
    assert missing['observed_cycle'] == 32
    assert missing['point_rul'] == prior['point_rul']


def test_missing_first_packet_withholds_forecast(client):
    value = client.post('/api/predict/rul', json={'engine_id':20, 'cycle':30, 'packet_loss':True}).json()
    assert value['point_rul'] is None and value['interval'] is None
    assert value['quality'] == 'STALE'


def test_offset_is_scored_but_requires_review(client):
    base = client.post('/api/predict/rul', json={'engine_id':20, 'cycle':60}).json()
    shift = client.post('/api/predict/rul', json={'engine_id':20, 'cycle':60, 'offsets':{'s4':30}}).json()
    assert shift['alert_state'] == 'REVIEW' and shift['quality'] == 'SHIFT_REVIEW'
    assert not shift['interval']['clean_benchmark_request']
    assert not np.isclose(base['point_rul'], shift['point_rul'])


def test_policy_is_independent_of_other_requests(client):
    body = {'engine_id':20, 'cycle':60}
    before = client.post('/api/predict/rul', json=body).json()
    client.post('/api/predict/rul', json={'engine_id':81, 'cycle':30, 'offsets':{'s4':20}})
    after = client.post('/api/predict/rul', json=body).json()
    for key in ['point_rul','risk_aware_rul','interval','alert_state','attribution']:
        assert before[key] == after[key]


def test_custom_prefix_matches_replay(client):
    raw = pd.read_csv(ROOT/'data/demo_slice/cmapss_replay.csv')
    rows = raw[(raw.unit == 20) & (raw.cycle <= 60)].to_dict('records')
    replay = client.post('/api/predict/rul', json={'engine_id':20,'cycle':60}).json()
    custom = client.post('/api/predict/rul', json={'observations':rows}).json()
    assert np.isclose(replay['point_rul'], custom['point_rul'])


@pytest.mark.parametrize('body,status', [
    ({'engine_id':999},404), ({'engine_id':1,'cycle':9999},422),
    ({'horizon':-1},422), ({'offsets':{'not_a_sensor':10}},422),
    ({'offsets':{'s4':101}},422), ({'observations':[{}]*29},422),
    ({'observations':[{}]*401},422), ({'unexpected':'field'},422)])
def test_invalid_rul_requests(client,body,status):
    assert client.post('/api/predict/rul', json=body).status_code == status


def test_diagnosis_uses_physical_inputs_and_multilabel_outputs(client):
    samples = client.get('/api/engines').json()['manufacturing_samples']
    value = client.post('/api/diagnose/ai4i', json=samples[0]['telemetry']).json()
    assert value['provenance'] == 'live_model'
    assert set(value['modes']) == {'TWF','HDF','PWF','OSF','RNF'}
    assert 0 <= value['failure_probability'] <= 1
    assert all(0 <= p <= 1 for p in value['modes'].values())
    assert not value['probabilities_calibrated']
    assert client.post('/api/diagnose/ai4i', json={'torque':-1}).status_code == 422
    assert client.post('/api/diagnose/ai4i', json={'type':'X'}).status_code == 422
    assert client.post('/api/diagnose/ai4i', json={'TWF':1}).status_code == 422


def test_unavailable_runtime_fails_closed(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get('/health').status_code == 503
        assert client.post('/api/predict/rul', json={}).status_code == 503


def test_modified_artifact_is_rejected_before_deserialization(tmp_path):
    (tmp_path/'models').mkdir()
    (tmp_path/'models/manifest.json').write_text(json.dumps({'files':{'models/untrusted.joblib':{'sha256':'0'*64}}}))
    (tmp_path/'models/untrusted.joblib').write_bytes(b'changed')
    with pytest.raises(ValueError, match='verification failed'):
        verify_manifest(tmp_path)


def test_packet_loss_uses_causal_hold_last_inputs(client):
    from timeguard.features import COLUMNS
    raw = pd.read_csv(ROOT/'data/demo_slice/cmapss_replay.csv')
    prefix = raw[(raw.unit == 20) & (raw.cycle <= 32)].copy()
    prefix.loc[prefix.cycle % 3 == 0, COLUMNS[2:]] = np.nan
    prefix[COLUMNS[2:]] = prefix[COLUMNS[2:]].ffill()
    expected = client.post('/api/predict/rul', json={'observations':prefix.to_dict('records')}).json()
    actual = client.post('/api/predict/rul', json={'engine_id':20,'cycle':32,'packet_loss':True}).json()
    assert np.isclose(expected['point_rul'], actual['point_rul'])
    assert not actual['interval']['clean_benchmark_request']
