import numpy as np
import pandas as pd

SENSORS = [f's{i}' for i in [2,3,4,7,8,9,11,12,13,14,15,17,20,21]]
COLUMNS = ['unit','cycle','setting_1','setting_2','setting_3']+[f's{i}' for i in range(1,22)]


def read_cmapss(path):
    frame = pd.read_csv(path, sep=r'\s+', header=None)
    if frame.shape[1] != 26:
        raise ValueError(f'Expected 26 columns, found {frame.shape[1]}')
    frame.columns = COLUMNS
    if not np.isfinite(frame.to_numpy()).all() or frame.duplicated(['unit','cycle']).any():
        raise ValueError('Invalid numeric telemetry or duplicate cycle')
    for _, g in frame.groupby('unit', sort=False):
        if not np.all(np.diff(g.cycle.to_numpy()) == 1):
            raise ValueError('Cycles must be contiguous and ordered within each unit')
    return frame


def make_features(frame, window=30):
    """Only observed prefixes. No lifespan, target or future values enter features."""
    if window < 2:
        raise ValueError('window must be at least two')
    rows=[]
    time=np.arange(window, dtype=float)
    centered=time-time.mean()
    denom=np.dot(centered,centered)
    for unit,g in frame.groupby('unit', sort=True):
        z=g[SENSORS].ewm(alpha=.25, adjust=False).mean().to_numpy()
        cycles=g.cycle.to_numpy()
        settings=g[['setting_1','setting_2','setting_3']].to_numpy()
        for t in range(window-1,len(g)):
            a=z[t-window+1:t+1]
            row={'unit':int(unit),'cycle':int(cycles[t])}
            moments=[a[-1],a.mean(0),a.std(0,ddof=1),a.min(0),a.max(0),centered@a/denom]
            for name,values in zip(['last','mean','std','min','max','slope'],moments):
                row.update({f'{s}_{name}':float(v) for s,v in zip(SENSORS,values)})
            row.update({f'setting_{i+1}':float(v) for i,v in enumerate(settings[t])})
            rows.append(row)
    return pd.DataFrame(rows)


def ai4i_features(frame):
    out=frame[['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']].copy()
    if not frame.Type.isin(['L','M','H']).all():
        raise ValueError('Unknown AI4I type')
    for t in ['L','M','H']:
        out['Type_'+t]=(frame.Type==t).astype(int)
    out['temperature_gap']=out['Process temperature [K]']-out['Air temperature [K]']
    out['power_W']=out['Torque [Nm]']*out['Rotational speed [rpm]']*2*np.pi/60
    out['wear_load']=out['Tool wear [min]']*out['Torque [Nm]']
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError('Invalid AI4I physical inputs')
    return out
