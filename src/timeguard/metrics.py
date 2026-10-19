import numpy as np


def nasa_score(y, prediction):
    d=np.asarray(prediction)-np.asarray(y)
    return float(np.expm1(np.where(d<0,-d/13,d/10)).sum())


def rmse(y, prediction):
    return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(prediction))**2)))
