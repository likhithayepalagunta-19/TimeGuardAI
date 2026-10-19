import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from timeguard.conformal import correction
from timeguard.features import make_features, SENSORS
from timeguard.policy import MaintenancePolicy
from timeguard.metrics import nasa_score


def test_prefix_invariance():
    frame=pd.DataFrame({'unit':[1]*40,'cycle':range(1,41),'setting_1':0.,'setting_2':0.,'setting_3':0.})
    for i,s in enumerate(SENSORS):
        frame[s]=np.arange(40)+i
    full=make_features(frame)
    prefix=make_features(frame.iloc[:35])
    pd.testing.assert_frame_equal(full[full.cycle<=35].reset_index(drop=True),prefix)


def test_conformal_rank_and_small_sample():
    y=np.arange(1.,21.)
    assert correction(y,np.zeros(20),np.zeros(20))==19
    assert np.isinf(correction([1.],[0.],[0.]))


def test_late_error_costs_more_in_nasa_score():
    assert nasa_score([50],[60])>nasa_score([50],[40])


def test_policy_persistence_missing_and_recovery():
    p=MaintenancePolicy()
    assert [p.update(20) for _ in range(3)]==['OBSERVE','OBSERVE','PLAN_MAINTENANCE']
    assert p.update(20,valid=False)=='REVIEW'
    assert p.update(100)=='REVIEW'
    for _ in range(3):
        p.update(100)
    assert p.update(100)=='OBSERVE'
    p.state='PLAN_MAINTENANCE'
    for _ in range(8):
        assert p.update(50)=='PLAN_MAINTENANCE'
