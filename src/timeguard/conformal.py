import math
import numpy as np


def correction(y, lower, upper, alpha=0.1):
    """Nonnegative CQR correction; one independent endpoint per calibration unit."""
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between zero and one')
    y, lower, upper = map(np.asarray, (y, lower, upper))
    if len(y) == 0 or y.shape != lower.shape or y.shape != upper.shape:
        raise ValueError('nonempty, matching calibration arrays required')
    if not all(np.isfinite(a).all() for a in (y, lower, upper)):
        raise ValueError('calibration arrays must be finite')
    lo, hi = np.minimum(lower, upper), np.maximum(lower, upper)
    scores = np.maximum.reduce([lo-y, y-hi, np.zeros_like(y)])
    rank = math.ceil((len(y)+1)*(1-alpha))
    return float(np.sort(scores)[rank-1]) if rank <= len(y) else float('inf')


def intervals(lower, upper, qhat):
    lo, hi = np.minimum(lower, upper), np.maximum(lower, upper)
    return np.maximum(0, lo-qhat), hi+qhat
