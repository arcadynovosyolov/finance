# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 00:20:29 2017

@author: 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ====================================================================
# trying optimization
# ====================================================================

a = 2


def func_to_min(x):
    """ """
    return (x ** 2).sum()


def func_const(x):
    """ """
    return np.dot(x, a) - 1


# ====================================================================
# shells
# ====================================================================


def typical_shell(numbs=[0, 1, 4], hor=1, n_trials=1000,
                  variance_axis=True, short_long='both'):
    """
    short_long = 'long', 'short' or 'both'
    """
    df = read_dj_data('2008-01-01')
    rets = calc_returns(df)
    cl = rets.columns
    m = rets.mean() * hor
    c = rets.cov() * hor

    # numbs = [0, 1, 4]

    # nass = len(numbs)
    cls = cl[numbs]
    mt = m[cls]
    ct = c.loc[cls, cls]
    out_long = simulate_portfolios(ct, mt, n_trials=n_trials, long_only=True)
    out_short = simulate_portfolios(ct, mt, n_trials=n_trials)
    out_sharpe = sharpe_portfolio(ct, mt)

    if variance_axis:
        if short_long != 'long':
            plt.scatter(out_short[:, 1], out_short[:, 0], s=1)
        if short_long != 'short':
            plt.scatter(out_long[:, 1], out_long[:, 0], s=1, c='y')
        plt.scatter(out_sharpe[1], out_sharpe[0], c='r')
        plt.xlabel('Portfolio variance')
    else:
        if short_long != 'short':
            plt.scatter(out_long[:, 2], out_long[:, 0], s=1, c='g')
        if short_long != 'long':
            plt.scatter(out_short[:, 2], out_short[:, 0], s=1)
        plt.scatter(out_sharpe[2], out_sharpe[0], c='r')
        plt.xlabel('Portfolio standard deviation')
    plt.ylabel('Portfolio return')
    plt.title(', '.join(cls))

    return out_long, out_short, out_sharpe

# ====================================================================
# portfolios
# ====================================================================


def sharpe_portfolio(c, m):
    """ """
    w = 100 * minimize_sharpe_ratio(c, m)
    sm = np.dot(w, m)
    sv = vCv(c, w)
    ss = np.sqrt(sv)
    out = np.array([sm, sv, ss])
    return out


def simulate_portfolios(c, m, n_trials=500, long_only=False):
    """ """
    nass = len(m)
    if long_only:
        w = 100 * random_simplex_points(n_trials, nass)
    else:
        w = 100 * random_hypercube_points(n_trials, nass)

    # compute portfolios mean, variance and standard deviation
    for i in range(n_trials):
        wi = w[i, :]
        mp = np.dot(wi, m)
        vp = vCv(c, wi)
        sp = np.sqrt(vp)
        curr = np.array([mp, vp, sp]).reshape(1, 3)
        if i == 0:
            out = curr
        else:
            out = np.concatenate((out, curr))
    return out


# ====================================================================
# random points
# ====================================================================


def random_simplex_points(n, d):
    """ n d-dimensional simplex points
    """
    for i in range(n):
        wc = np.random.uniform(size=(d-1,))
        wc.sort()
        ww = np.concatenate((np.zeros((1,)), wc, np.ones((1,))))
        wc = np.diff(ww).reshape(1, d)
        if i == 0:
            w = wc
        else:
            w = np.concatenate((w, wc), axis=0)
    return w


def random_hypercube_points(n, d):
    """ n d-dimensional hypercube [-1, 1]^d points
    """
    w = 2 * np.random.uniform(size=(n, d)) - 1
    ws = w.sum(axis=1).repeat(d).reshape(n, d)
    ws[ws == 0] = 0.0000001
    return w / ws


# ====================================================================
# linear algebra stuff
# ====================================================================


def vCm1v(c, v):
    """ quadratic form with C^(-1), product v * C^(-1) * v
    """
    cv = np.linalg.solve(c, v)
    return v.dot(cv)


def vCv(c, v):
    """ quadratic form,  product v * C * v
    """
    return v.dot(c).dot(v)


# ====================================================================
# Sharpe ratio optimization stuff
# ====================================================================


def sharpe_optimize(c, m, long_only=False):
    """ """
    def sharpe_f(w):
        """ """
        return np.sqrt(vCv(c, w))

    def sharpe_eq(w):
        """ """
        return w.dot(m) - 1

    w0 = np.ones((len(m),))
    cons = ({'type': 'eq', 'fun': sharpe_eq})
    bnds = [(0, 1000000)] * len(m) if long_only else None
    mm = minimize(sharpe_f, w0, bounds=bnds, constraints=cons)
    ms = mm.x.sum()
    if ms != 0:
        return mm.x / ms
    else:
        return mm.x

# ====================================================================
# Sharpe ratio analytic stuff (short/long assets admissible)
# ====================================================================


def sharpe_fun(c, m, w):
    """ inverse of a standard Sharpe ratio """
    wcw = vCv(c, w)
    wm = w.dot(m)
    return np.sqrt(wcw) / wm


# ====================================================================
# preliminaries
# ====================================================================


def read_dj_data(start_date='2001-01-01', remove_gaps=True):
    """ """
    df = pd.read_csv('dj_quandl_2001-01-01.csv', index_col='Date')
    indl = (df.index >= start_date)
    dfn = df[indl]
    dfn.index = pd.to_datetime(dfn.index)
    if remove_gaps:
        df = remove_rows_with_gaps(dfn)
    df.columns = [x[5:] for x in df.columns]
    return df


def calc_returns(df, log_returns=False):
    """ """
    n = df.shape[0]
    rets = np.array(df.ix[1:n, :]) / np.array(df.ix[:(n-1), :])
    if log_returns:
        rets = np.log(rets)
    else:
        rets = rets - 1
    retsdf = pd.DataFrame(rets, index=df.index[:(n-1)], columns=df.columns)
    return retsdf


def remove_rows_with_gaps(df):
    """ remove all dates with missing quotes """
    ind = df.isnull()
    indi = ind.any(axis=1)
    return df.drop(df.index[indi])


def which_are_null(df):
    """ """
    cl = df.columns
    ind = df.isnull()
    su = ind.sum()
    li = list()
    for i in range(len(su)):
        if su[i] > 0:
            indi = ind.ix[:, i]
            ss = df.index[indi]
            print(cl[i], ss)
            li.append(ss)
    return li
