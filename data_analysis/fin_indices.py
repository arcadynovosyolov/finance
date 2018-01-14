# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 10:00:57 2017

@author: Arcady Novosyolov
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
# import datetime as dt

import quandl
quandl.ApiConfig.api_key = "wgEAn7VKLsTrwucXtRML"

# print(dt.date.today().isoformat())
vix_local = 'vix_index.csv'
snp_local = 'snp_index.csv'
dji_local = 'djia_index.csv'


class SnP500:
    """ """

    def __init__(self, start_date='1950-01-01', end_date='2030-12-31',
                 use_local=True):
        """ """
        if use_local and os.path.exists(snp_local):
            dfo = get_local_snp()[start_date:end_date]
            sf = '{0} observations from {1} to {2} read'
            print(sf.format(dfo.shape[0], dfo.index[0], dfo.index[-1]))
        else:
            dfo = get_yahoo_snp(st=start_date, en=end_date)
            if type(dfo) == str:
                print('Reading local data')
                dfo = get_local_snp()[start_date:end_date]
                sf = '{0} observations from {1} to {2} read'
                print(sf.format(dfo.shape[0], dfo.index[0], dfo.index[-1]))
            else:
                sf = '{0} observations from {1} to {2} downloaded'
                print(sf.format(dfo.shape[0], dfo.index[0], dfo.index[-1]))
        dfo['spread'] = (dfo.high - dfo.low) / dfo.high
        dfo['move'] = (dfo.close - dfo.open) / dfo.close
        self.df = dfo


class Vix:
    """ """

    def __init__(self, start_date='2004-01-01', end_date='2030-12-31',
                 use_local=True, hors=[], col='close'):
        """ adds returns for all fields, and a shift back by 'hor' days
        of the 'col' field
        """
        if use_local and os.path.exists(vix_local):
            dfo = get_Local_vix()[start_date:end_date]
            sf = '{0} observations from {1} to {2} read'
            print(sf.format(dfo.shape[0], dfo.index[0], dfo.index[-1]))
        else:
            dfo = get_quandl_vix(st=start_date, en=end_date)
            sf = '{0} observations from {1} to {2} downloaded'
            print(sf.format(dfo.shape[0], dfo.index[0], dfo.index[-1]))
        dfo['ret_close'] = (dfo.close.shift(-1) - dfo.close) / dfo.close
        dfo['ret_open'] = (dfo.open.shift(-1) - dfo.open) / dfo.open
        dfo['ret_high'] = (dfo.high.shift(-1) - dfo.high) / dfo.high
        dfo['ret_low'] = (dfo.low.shift(-1) - dfo.low) / dfo.low
        dfo['spread'] = (dfo.high - dfo.low) / dfo.high
        dfo['move'] = (dfo.close - dfo.open) / dfo.close
        for hor in hors:
            coln = '{0}_{1}'.format(col, hor)
            dfo[coln] = 100 * (dfo[col].shift(-hor) / dfo[col] - 1)
        self.hors = hors
        self.col = col
        self.df = dfo
        self.int_length = 5

    def select_data(self, interval=(10, 11)):
        """ select back data from a self.col range
        for studying conditional distribution
        """
        cl = self.col
        ind = (self.df[cl] >= interval[0]) & (self.df[cl] <= interval[1])
        dfs = self.df[ind]
        return dfs

    def cdfs(self, x, interval=None, printIt=False):
        """ cumulative dictribution functions of returns from the given
        interval over horizons at points 'x' as percentage returns
        """
        if interval is None:
            interval = (0, 100)
        cl = self.col
        ind = (self.df[cl] >= interval[0]) & (self.df[cl] <= interval[1])
        dfs = self.df[ind]
        if printIt:
            fs = '{0} observations for {1} between {2} and {3}'
            print(fs.format(dfs.shape[0], cl, interval[0], interval[1]))
        dfo = pd.DataFrame(index=x)
        dfo.index.name = 'return, %'
        for hor in self.hors:
            coln = '{0}_{1}'.format(self.col, hor)
            dfo[coln] = [(dfs[coln] <= xx).mean() for xx in x]
        return dfo

    def make_intervals_col(self, int_len=5):
        """ """
        self.int_length = int_len
        self.df['intervals'] = self.df.close.apply(self.str_interval)
        self.df['int_center'] = self.df.close.apply(self.str_interval_center)

    def str_interval(self, va):
        """ string representation of the interval for 'va' """
        mi = self.int_length * np.floor(va / self.int_length)
        ma = mi + self.int_length
        return '[{0}, {1})'.format(mi, ma)

    def str_interval_center(self, va):
        """ string representation of the interval for 'va' """
        mi = self.int_length * np.floor(va / self.int_length)
        ma = mi + self.int_length
        return 0.5 * (mi + ma)

# -----------------------------------------------------------------------
# VIX stuff
# -----------------------------------------------------------------------


def some_plots(inter=(10, 15)):
    """ """
    vix = Vix(hors=[15, 30, 60, 90])
    ret_cols = [s for s in vix.df.columns if s.startswith(vix.col + '_')]
    x = np.linspace(-30, 30, 25)
    dfo = vix.cdfs(x, interval=inter)
    for cl in ret_cols:
        dfo[cl].plot(label=cl)
    plt.legend()
    fs = 'CDFs of VIX returns, starting between {0} and {1}'
    plt.title(fs.format(inter[0], inter[1]))


def get_quandl_vix(st='2004-01-01', en='2030-12-31'):
    """ """
    df = quandl.get('cboe/vix', start_date=st, end_date=en)
    df.index.name = 'date'
    df.columns = ['open', 'high', 'low', 'close']
    save_local_vix(df)
    return df


# -----------------------------------------------------------------------
# DJIA stuff
# -----------------------------------------------------------------------


def get_yahoo_djia(st='1895-01-01', en='2030-12-31'):
    """ get DJIA data from Yahoo Finance """
    df = web.get_data_yahoo('^dji', start=st, end=en)
    df.columns = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
    df['ret'] = df.adj_close.shift(-1) / df.adj_close - 1
    df.index.name = 'date'
    save_local_dji(df)
    return df


# -----------------------------------------------------------------------
# S&P500 stuff
# -----------------------------------------------------------------------


def get_yahoo_snp(st='1950-01-01', en='2030-12-31'):
    """ get S&P500 data from Yahoo Finance """
    try:
        df = web.get_data_yahoo('^gspc', start=st, end=en)
        df.columns = ['open', 'high', 'low', 'close', 'volume', 'adj_close']
        df['ret'] = df.adj_close.shift(-1) / df.adj_close - 1
        df.index.name = 'date'
        save_local_snp(df)
    except:
        df = 'Error getting data from Yahoo'
        print(df)
    return df


# ========================================================
# local stuff
# ========================================================

def get_Local_vix():
    """ """
    df = pd.read_csv(vix_local).set_index('date')
    return df


def save_local_vix(df):
    """ """
    df.to_csv(vix_local)


def save_local_snp(df):
    """ """
    df.to_csv(snp_local)


def get_local_snp(st='1950-01-01', en='2030-12-31'):
    """ """
    df = pd.read_csv(snp_local).set_index('date')
    return df[st:en]


def save_local_dji(df):
    """ """
    df.to_csv(dji_local)


def get_Local_dji():
    """ """
    df = pd.read_csv(dji_local).set_index('date')
    return df
