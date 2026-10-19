"""AI4I-specific snapshot diagnosis. Never apply this model to turbofan data."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import pandas as pd
import streamlit as st
from timeguard.features import ai4i_features

st.set_page_config(page_title='TimeGuard · Manufacturing diagnosis',layout='wide')
st.title('Manufacturing diagnosis')
st.caption('Separate AI4I domain. Synthetic manufacturing benchmark, not turbofan failure diagnosis. Scores are uncalibrated model probabilities.')
if not (ROOT/'models/modes.joblib').exists():
    st.info('Run python scripts/train.py --task modes first.')
    st.stop()

@st.cache_resource
def load():
    return joblib.load(ROOT/'models/modes.joblib')

bundle=load()
with st.form('snapshot'):
    a,b,c=st.columns(3)
    typ=a.selectbox('Product quality type',['L','M','H'])
    air=b.number_input('Air temperature, K',value=300.,min_value=200.,max_value=500.)
    process=c.number_input('Process temperature, K',value=310.,min_value=200.,max_value=600.)
    rpm=a.number_input('Rotational speed, rpm',value=1500,min_value=1,max_value=10000)
    torque=b.number_input('Torque, Nm',value=40.,min_value=0.,max_value=200.)
    wear=c.number_input('Tool wear, minutes',value=100,min_value=0,max_value=500)
    submitted=st.form_submit_button('Analyze snapshot')
if submitted:
    frame=pd.DataFrame([{'Type':typ,'Air temperature [K]':air,'Process temperature [K]':process,'Rotational speed [rpm]':rpm,'Torque [Nm]':torque,'Tool wear [min]':wear}])
    x=ai4i_features(frame)[bundle['features']]
    p=float(bundle['binary'].predict_proba(x)[0,1])
    st.metric('Failure score, uncalibrated',f'{p:.1%}')
    rows=[]
    for name,model in bundle['modes'].items():
        score=float(model.predict_proba(x)[0,list(model.classes_).index(1)]) if 1 in model.classes_ else 0.
        rows.append({'mode':name,'uncalibrated_score':score,'threshold_0_5':score>=.5})
    st.dataframe(pd.DataFrame(rows),hide_index=True)
    st.warning('Mode labels can overlap. The current holdout misses TWF and RNF positives at threshold 0.5. Do not interpret low scores as proof of safe operation. Real factory use requires new validation.')
