# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 20:34:20 2018

@author: Arcady Novosyolov
"""

import numpy as np
import pandas as pd
import gaussian_copula_stuff as gcs
import time as tt

# ==========================================================
# create all models for a period
# ==========================================================

def all_models(start_date='2017-01-01',
               end_date='2017-06-30',
               verbose=1):
    """ """
    # basic data: returns from 2010 till 2017
    dfi = pd.read_csv('indices.csv').set_index('date')
    dfs = pd.read_csv('stocks.csv').set_index('date')
    indices = dfi.columns
    stocks = dfs.columns
    df = dfi.join(dfs, how='inner')
    
    dfi = df[indices]
    dfs = df[stocks]
    
    # select dates
    indd = (df.index >= start_date) & (df.index <= end_date)
    dates = df.index[indd]
    
    start_time = tt.time()
    models = pd.Series()
    for dat in dates:
        mvd = gcs.MultiVarDistribution()
        mvd.fit(df)
        models[dat] = mvd
        if verbose >= 1:
            show_time(start_time, txt='model for {0} built'.format(dat))
    frm = 'models for dates from {0} to {1} have been built'
    print(frm.format(start_date, end_date))
    show_time(start_time, txt='total process')
    return models, dfi, dfs

# ==========================================================
# strategies
# ==========================================================

def create_distributions(models,                   # all models
                         dfi,                      # all indices data
                         dfs):                     # all stocks data
    """ """
    start_time = tt.time()
    dates = models.index
    df = pd.DataFrame()
    for dat in dates:
        mvd = models[dat]
        se = pd.Series()
        x_cond = dfi.loc[[dat], :]
        for stock in dfs.columns:
            x = dfs.loc[[dat], [stock]]
            se[stock] = x.iloc[0, 0]
            se[stock+'_marg'] = mvd.marg_cdf(x)[0]
            se[stock+'_cond'] = mvd.cond_cdf(x, x_cond)[0]
        se.name = dat
        df = df.append(se)
    # convert returns to cumulative returns
    for stock in dfs.columns:
        df[stock] = (1 + df[stock]).cumprod()
    show_time(start_time, txt='creating all distributions')
    return df

def create_strategy(stock, df, buy_value=0.2, sell_value=0.8):
    """
    df taken from 'create_distributions' function
    """
    # select columns just for 'stock'
    cols = [s for s in df.columns if s.startswith(stock)]
    dfp = df[cols]
    
    # find conditional distribution column
    cond_col = [s for s in dfp.columns if s.endswith('_cond')][0]
    cond_distr = dfp[cond_col]
    
    # prepare a series for trading signals
    se = pd.Series(0, index=df.index)
    se[cond_distr <= buy_value] = 1
    se[cond_distr >= sell_value] = -1
    
    # remove repeated buys and repeated sells
    se.name = 'main'
    sed = se.diff()
    sed.name = 'dif'
    
    sen = se[se.nonzero()[0]]
    sed = sen.diff()
    sed.name = 'dif'
    se = (sed.fillna(value=0) / 2).astype('int')
    
    # complete the sequance, if there are open position
    su = se.sum()
    if su != 0:
        if se.index[-1] == df.index[-1]:
            # cancel last day operation
            se.drop(se.index[-1], inplace=True)
        else:
            # appens the last day closing operation
            se[df.index[-1]] = -su
    ind = (se != 0)
    se = se[ind]
    # compute return
    dfo = df[[stock]].join(se, how='inner')
    st = dfo[stock]
    dfo['rat'] = st.shift(-1) / st
    dfo['ratt'] = dfo.rat ** dfo.dif
    return dfo.dropna().ratt.prod()

def create_strategies(stocks, df, buy_value=0.2, sell_value=0.8):
    """
    df taken from 'create_distributions' function
    """
    se = pd.Series(index=stocks)
    for stock in stocks:
        se[stock] = create_strategy(stock, df,
          buy_value=buy_value, sell_value=sell_value)
    return se

# ==========================================================
# auxiliary functions
# ==========================================================

def show_time(start_time, txt=''):
    """ """
    pre = txt + ', ' if len(txt) > 0 else ''
    frm = pre + 'elapsed time = {0:.3f} seconds'
    print(frm.format(tt.time() - start_time))
