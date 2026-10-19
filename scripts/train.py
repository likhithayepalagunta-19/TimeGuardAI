"""CPU benchmark training. Outputs actual measurements, never example scores."""
import argparse
import json
import sys
import platform
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from timeguard.features import read_cmapss, make_features, ai4i_features
from timeguard.conformal import correction, intervals
from timeguard.metrics import nasa_score, rmse


def save_report(name, result):
    result['environment']={'python':platform.python_version(),'sklearn':sklearn.__version__,'cpu':platform.processor(),'seed':42}
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports'/f'{name}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2))


def rul():
    raw=read_cmapss(ROOT/'data/raw/cmapss/train_FD001.txt')
    feats=make_features(raw)
    final_cycles=raw.groupby('unit').cycle.max()
    feats['raw_rul']=feats.unit.map(final_cycles)-feats.cycle
    names=[c for c in feats if c not in ['unit','cycle','raw_rul']]
    units=np.sort(feats.unit.unique())
    fit_ids,other=train_test_split(units,test_size=.4,random_state=42)
    cal_ids,val_ids=train_test_split(other,test_size=.5,random_state=42)
    fit=feats[feats.unit.isin(fit_ids)]
    rng=np.random.default_rng(42)
    def endpoints(ids):
        rows=[]
        for unit in sorted(ids):
            g=feats[feats.unit==unit]
            # One preregistered prefix endpoint per unit, not overlapping windows.
            observed=int(final_cycles.loc[unit]*rng.uniform(.6,.95))
            eligible=g[g.cycle<=max(30,observed)]
            rows.append(eligible.iloc[-1])
        return pd.DataFrame(rows)
    cal,val=endpoints(cal_ids),endpoints(val_ids)
    models={}
    for tau in [.05,.2,.5,.95]:
        model=GradientBoostingRegressor(loss='quantile',alpha=tau,n_estimators=100,max_depth=3,learning_rate=.06,random_state=42)
        model.fit(fit[names],fit.raw_rul.clip(upper=125))
        models[tau]=model
        print('Fitted quantile',tau,flush=True)
    qhat=correction(cal.raw_rul.to_numpy(),models[.05].predict(cal[names]),models[.95].predict(cal[names]))
    baseline=GradientBoostingRegressor(n_estimators=100,max_depth=3,random_state=42)
    baseline.fit(fit[names],fit.raw_rul.clip(upper=125))
    test=read_cmapss(ROOT/'data/raw/cmapss/test_FD001.txt')
    test_features=make_features(test).groupby('unit',sort=True).tail(1).sort_values('unit')
    official=np.loadtxt(ROOT/'data/raw/cmapss/RUL_FD001.txt').reshape(-1)
    all_units=np.sort(test.unit.unique())
    if len(all_units)!=len(official):
        raise ValueError('Test unit count does not match RUL labels')
    truth=dict(zip(all_units,official))
    y=test_features.unit.map(truth).to_numpy()
    x=test_features[names]
    point=models[.5].predict(x)
    decision=models[.2].predict(x)
    lo,hi=intervals(models[.05].predict(x),models[.95].predict(x),qhat)
    for _ in range(10):
        models[.5].predict(x.iloc[:1])
    times=[]
    for _ in range(100):
        t=time.perf_counter()
        # Full prediction bundle, not merely the median estimator.
        for model in models.values():
            model.predict(x.iloc[:1])
        times.append((time.perf_counter()-t)*1000)
    def stats(pred):
        return {'rmse_clipped':rmse(np.minimum(y,125),pred),'rmse_raw':rmse(y,pred),'nasa_score_raw':nasa_score(y,pred),'late_rate':float(np.mean(pred>y))}
    bundle={'models':models,'features':names,'qhat':qhat,'window':30,'sensors':list(raw.columns),'fit_ids':fit_ids.tolist(),'cal_ids':cal_ids.tolist(),'val_ids':val_ids.tolist()}
    (ROOT/'models').mkdir(exist_ok=True)
    joblib.dump(bundle,ROOT/'models/rul.joblib')
    vlo,vhi=intervals(models[.05].predict(val[names]),models[.95].predict(val[names]),qhat)
    save_report('rul',{'test_engine_count':len(y),'excluded_short_engines':len(all_units)-len(y),'calibration_engine_count':len(cal),'qhat_raw_rul':qhat,'median':stats(point),'decision_tau_020':stats(decision),'baseline':stats(baseline.predict(x)),'coverage_90_raw':float(np.mean((lo<=y)&(y<=hi))),'mean_width':float(np.mean(hi-lo)),'validation_coverage_raw':float(np.mean((vlo<=val.raw_rul)&(val.raw_rul<=vhi))),'bundle_model_only_ms_p50':float(np.percentile(times,50)),'bundle_model_only_ms_p95':float(np.percentile(times,95)),'protocol':'One official final endpoint per eligible test engine; calibration endpoints at random 60-95 percent lifetime. Model trained on capped RUL; interval calibrated against raw RUL.'})


def modes():
    frame=pd.read_csv(ROOT/'data/raw/ai4i/ai4i2020.csv')
    x=ai4i_features(frame)
    mode_names=['TWF','HDF','PWF','OSF','RNF']
    fit_idx,test_idx=train_test_split(np.arange(len(frame)),test_size=.2,stratify=frame['Machine failure'],random_state=42)
    binary=RandomForestClassifier(n_estimators=250,min_samples_leaf=2,class_weight='balanced_subsample',random_state=42,n_jobs=-1)
    binary.fit(x.iloc[fit_idx],frame['Machine failure'].iloc[fit_idx])
    classifiers={}
    results={}
    def measure(y,p):
        predicted=(p>=.5).astype(int)
        pr,re,f1,_=precision_recall_fscore_support(y,predicted,average='binary',zero_division=0)
        return {'precision':float(pr),'recall':float(re),'f1':float(f1),'pr_auc':float(average_precision_score(y,p)) if np.any(y) else None,'positive_support':int(np.sum(y))}
    results['binary']=measure(frame['Machine failure'].iloc[test_idx],binary.predict_proba(x.iloc[test_idx])[:,1])
    for name in mode_names:
        clf=RandomForestClassifier(n_estimators=200,min_samples_leaf=2,class_weight='balanced_subsample',random_state=42,n_jobs=-1)
        clf.fit(x.iloc[fit_idx],frame[name].iloc[fit_idx])
        probabilities=clf.predict_proba(x.iloc[test_idx])
        p=probabilities[:,list(clf.classes_).index(1)] if 1 in clf.classes_ else np.zeros(len(test_idx))
        results[name]=measure(frame[name].iloc[test_idx],p)
        classifiers[name]=clf
    joblib.dump({'binary':binary,'modes':classifiers,'features':list(x.columns),'test_indices':test_idx.tolist()},ROOT/'models/modes.joblib')
    save_report('modes',{'split':'Stratified binary random 80/20; generated-row dependence remains a limitation','threshold':.5,'labels':results,'rn_limit':'Random failures are not physically predictable from these inputs.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=['rul','modes'],required=True)
    args=parser.parse_args()
    (ROOT/'models').mkdir(exist_ok=True)
    rul() if args.task=='rul' else modes()
