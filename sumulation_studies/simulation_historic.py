# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 14:50:14 2017

@author: Arcady Novosyolov
"""

import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt

use_cyrillic = False
# use_cyrillic = True

if use_cyrillic:
    failed_txt = 'Отвергается'
    fig_columns = ['А', 'Б', 'В']
    asset_title = 'Актив'
else:
    failed_txt = 'Rejected'
    fig_columns = ['A', 'B', 'C']
    asset_title = 'Asset'

hyp_txt = {False: failed_txt, True: '-'}

# ===============================================================
# shells
# ===============================================================


def main_shell():
    """ """
    df = read_data()
    dfr = calc_all_returns(df)
    autocorr_all(dfr, depth=21).loc[1:, :].plot()
    dfh = direct_hist_sample(dfr)
    autocorr_all(dfh, depth=21).loc[1:, :].plot()
    dfb = bunched_hist_sample(dfr)
    autocorr_all(dfb, depth=21).loc[1:, :].plot()

# ===============================================================
# reading the data
# ===============================================================


def read_data():
    """ """
    df = pd.read_csv('sdnmagv.csv')
    df['date'] = pd.to_datetime(df.date)
    return df.set_index('date')


# ===============================================================
# calculating returns from quates
# ===============================================================


def calc_returns(se, remove_na=True):
    """ calculating returns from a single Series """
    ser = se.shift(-1) / se - 1
    if remove_na:
        ser.dropna(inplace=True)
    return ser


def calc_all_returns(df, remove_na=True):
    """ calculating returns from a DataFrame of quotes """
    df_ret = pd.DataFrame(index=df.index)
    for cl in df.columns:
        df_ret[cl] = calc_returns(df[cl], False)
    if remove_na:
        df_ret.dropna(inplace=True)
    return df_ret


def calc_some_returns(df, cls, remove_na=True):
    """ calculting returns from a DataFrame for selected columns 'cls' only """
    df_ret = pd.DataFrame(index=df.index)
    for cl in cls:
        df_ret[cl] = calc_returns(df[cl], False)
    if remove_na:
        df_ret.dropna(inplace=True)
    return df_ret

# ===============================================================
# autocorrelations
# ===============================================================


def autocorr(se, depth=21):
    """ autocorrelations from a single Series """
    dfa = pd.DataFrame(se)
    for i in range(depth):
        dfa['z{0}'.format(i + 1)] = se.shift(i + 1)
    cr = dfa.corr()
    cr.index = range(depth + 1)
    return cr[se.name]


def autocorr_all(df, depth=21):
    """ autocorrelations from a DataFrame, for all columns """
    dfa = pd.DataFrame()
    for cl in df.columns:
        dfa[cl] = autocorr(df[cl], depth=depth)
    return dfa

# ===============================================================
# Confidence levels for E(XY) given zero correlation
# ===============================================================


def mus_with_bounds(dfr, cl='apple', depth=5, q=ss.norm.ppf(0.975)):
    """ """
    se = dfr[cl]
    sig2 = se.var()
    dft = dfr[[cl]].copy()
    lst = list()
    for i in range(1, depth + 1):
        clsh = '{0}'.format(i)
        lst.append(clsh)
        dft[clsh] = se.shift(-i)
    dft.dropna(inplace=True)
    n_loc = dft.shape[0]
    bnd = q * sig2 / n_loc ** 0.5
    dfo = pd.DataFrame(index=['lower', 'mu_hat', 'upper'])
    for clsh in lst:
        mu_hat = (dft[cl] * dft[clsh]).mean()
        dfo[clsh] = [-bnd, mu_hat, bnd]
    return dfo.T / sig2


def plot_with_bounds(dfr, num=0):
    """ """
    tick = dfr.columns[num]
    dfo = mus_with_bounds(dfr, cl=tick)
    dfo.plot()
    plt.title(tick)
    plt.xlabel('lag')
    plt.ylabel('correlation')

# ===============================================================
# Historic sampling
# ===============================================================


def direct_hist_sample(dfr, n_trials=5000):
    """ sampling returns one by one """
    ind = np.random.choice(dfr.index, size=n_trials)
    dfh = dfr.loc[ind, :]
    dfh.index = range(dfh.shape[0])
    return dfh


def bunched_hist_sample(dfr, n_trials=5000, bunchsize=20):
    """
    sampling return by bunches of 'bunchsize' returns in a row,
    with no overlapping allowed
    """
    # convert to array to avoid indexing trouble
    aa = np.array(dfr)

    # number of bunches in the dataframe dfr
    nb = dfr.shape[0] // bunchsize

    # number of bunches to generate
    nbg = n_trials // bunchsize

    # numbers of bunches selected for sampling
    bind = np.random.choice(range(nb), size=nbg)

    # indices of records for sampling
    ind = np.zeros((nbg, bunchsize))

    # fill indices of records
    for i in range(nbg):
        st = bind[i] * bunchsize
        ind[i, :] = range(st, st + bunchsize)

    # convert indices to integer type
    indr = ind.reshape((nbg * bunchsize, )).astype('int')

    # make actual sampling
    aaa = aa[indr, :]

    # make up a dataframe for output
    dfb = pd.DataFrame(aaa, columns=dfr.columns)
    return dfb


def bunched_overlap_hist_sample(dfr, n_trials=5000, bunchsize=20):
    """ bunches with overlapping """
    # convert to array to avoid indexing trouble
    aa = np.array(dfr)

    # number of bunches in the dataframe dfr (different from non-ovr case)
    nb = dfr.shape[0] - bunchsize

    # number of bunches to generate
    nbg = n_trials // bunchsize

    # numbers of bunches selected for sampling
    bind = np.random.choice(range(nb), size=nbg)

    # indices of records for sampling
    ind = np.zeros((nbg, bunchsize))

    # fill indices of records
    for i in range(nbg):
        ind[i, :] = range(bind[i], bind[i] + bunchsize)

    # convert indices to integer type
    indr = ind.reshape((nbg * bunchsize, )).astype('int')

    # make actual sampling
    aaa = aa[indr, :]

    # make up a dataframe for output
    dfb = pd.DataFrame(aaa, columns=dfr.columns)
    return dfb
