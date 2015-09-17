# pylint: disable=missing-docstring, invalid-name
import numpy as np

def get_polynomial(xs, ys, degree):
    """Return a numpy polynomial and r2 value."""
    x = np.array(list(xs))
    y = np.array(list(ys))
    return np.polyfit(x, y, degree)

def get_r2_from_poly(xs, ys, poly):
    x = np.array(list(xs))
    y = np.array(list(ys))
    yfit = np.polyval(poly, x)
    return _get_r2(y, yfit)

def get_r2_from_model(xs, ys, model):
    np_xs = np.array(list(xs))
    np_ys = np.array(list(ys))
    yfit = np.array([model(x) for x in np_xs])
    return _get_r2(np_ys, yfit)

def _get_r2(y, yfit):
    yresid = y - yfit
    ss_resid = sum(np.power(yresid, 2))
    ss_total = len(y) * y.var()
    return 1 - ss_resid/ss_total
