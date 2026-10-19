import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import shap
import pandas as pd
from timeguard.features import read_cmapss, make_features

parser=argparse.ArgumentParser()
parser.add_argument('--engine',type=int,default=1)
args=parser.parse_args()
bundle=joblib.load(ROOT/'models/rul.joblib')
raw=read_cmapss(ROOT/'data/raw/cmapss/test_FD001.txt')
x=make_features(raw[raw.unit==args.engine]).tail(1)[bundle['features']]
values=shap.TreeExplainer(bundle['models'][.5])(x).values[0]
result=pd.DataFrame({'feature':x.columns,'observed_value':x.iloc[0].to_numpy(),'shap_rul_cycles':values})
result['sensor']=result.feature.str.replace(r'_(last|mean|std|min|max|slope)$','',regex=True)
result['abs_shap']=result.shap_rul_cycles.abs()
(ROOT/'reports').mkdir(exist_ok=True)
result.sort_values('abs_shap',ascending=False).to_csv(ROOT/'reports'/f'shap_engine_{args.engine}.csv',index=False)
print(result.sort_values('abs_shap',ascending=False).head(12).to_string(index=False))
