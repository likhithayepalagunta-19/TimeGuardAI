"""Actual benchmark RUL replay. Requires a locally trained model bundle."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from timeguard.features import read_cmapss, make_features
from timeguard.conformal import intervals
from timeguard.policy import MaintenancePolicy

st.set_page_config(page_title='TimeGuard AI',page_icon='⏱',layout='wide')
st.markdown('<style>.stApp{background:#faf9fc}h1,h2,h3{color:#66567d}div[data-testid="stMetric"]{background:#eae5f5;padding:20px;border-radius:14px}</style>',unsafe_allow_html=True)
st.title('TimeGuard AI · Engine replay')
st.caption('NASA C-MAPSS FD001 benchmark replay. Cycles are not wall-clock hours. Human maintenance review required.')
if not (ROOT/'models/rul.joblib').exists():
    st.info('Train the real RUL bundle first: python scripts/download_data.py --dataset cmapss, then python scripts/train.py --task rul. For the synthetic Stage 1 demo, open portal/index.html.')
    st.stop()

@st.cache_resource
def load_bundle():
    return joblib.load(ROOT/'models/rul.joblib')

@st.cache_data
def load_data():
    return read_cmapss(ROOT/'data/raw/cmapss/test_FD001.txt')

bundle=load_bundle()
data=load_data()
engine=st.sidebar.selectbox('Engine',sorted(data.unit.unique()))
asset=data[data.unit==engine]
limit=int(asset.cycle.max())
if limit<30:
    st.warning('This engine has fewer than 30 cycles. Model abstains during warmup.')
    st.stop()
horizon=st.sidebar.slider('Maintenance horizon, cycles',10,100,45)
speed=st.sidebar.select_slider('Replay interval, seconds',options=[.25,.5,1.,2.],value=1.)
playing=st.sidebar.toggle('Play replay')
if st.sidebar.button('Reset') or st.session_state.get('engine')!=int(engine):
    st.session_state.engine=int(engine)
    st.session_state.cursor=30
st.session_state.setdefault('cursor',30)

@st.fragment(run_every=speed if playing else None)
def replay():
    if playing:
        st.session_state.cursor=min(limit,st.session_state.cursor+1)
    if st.button('Step one cycle'):
        st.session_state.cursor=min(limit,st.session_state.cursor+1)
    cursor=st.session_state.cursor
    prefix=asset[asset.cycle<=cursor]
    features=make_features(prefix)
    x=features[bundle['features']]
    model=bundle['models']
    point=model[.5].predict(x)
    decision=model[.2].predict(x)
    lo,hi=intervals(model[.05].predict(x),model[.95].predict(x),bundle['qhat'])
    policy=MaintenancePolicy(horizon=horizon)
    states=[policy.update(float(v),np.isfinite(v)) for v in lo]
    a,b,c=st.columns(3)
    a.metric('Median RUL',f'{point[-1]:.1f} cycles')
    b.metric('Nominal 90% prediction interval',f'{lo[-1]:.0f}–{hi[-1]:.0f}')
    c.metric('Policy',states[-1].replace('_',' ').title())
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=features.cycle,y=hi,line_color='#c0dccf',name='Upper'))
    fig.add_trace(go.Scatter(x=features.cycle,y=lo,fill='tonexty',fillcolor='rgba(183,219,204,.3)',line_color='#c0dccf',name='Lower'))
    fig.add_trace(go.Scatter(x=features.cycle,y=point,line_color='#a28abd',name='Median RUL'))
    fig.add_trace(go.Scatter(x=features.cycle,y=decision,line_color='#d5a58d',name='Decision RUL, τ=.20'))
    fig.add_hline(y=horizon,line_dash='dash',line_color='#c7a285')
    fig.update_layout(template='plotly_white',xaxis_title='Observed cycle',yaxis_title='Remaining cycles',height=400)
    st.plotly_chart(fig,use_container_width=True)
    st.caption(f'Observed through cycle {cursor}/{limit}. Empirical coverage is in reports/rul.json. Calibration assumptions can fail under shift. This replay uses clean benchmark data, with no injected-fault validation claim.')
    evidence=pd.DataFrame({'cycle':features.cycle.to_numpy(),'median_rul':point,'decision_rul':decision,'lower':lo,'upper':hi,'policy':states})
    st.download_button('Download observed evidence',evidence.to_csv(index=False),'timeguard-model-evidence.csv','text/csv')
    if cursor==limit:
        st.success('End of observed test prefix. Future measurements remain hidden.')
replay()
