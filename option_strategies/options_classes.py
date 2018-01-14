# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 15:58:25 2017

@author: 1
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import UserList
# import matplotlib.pyplot as plt

import string

# now lets see what this string module provide us.
# I wont be going into depth because the python
# documentation provides ample information.
# so lets generate a random string with 32 characters.

use_cyrillic = False
# use_cyrillic = True


def random_string(s_len):
    """ random string from ascii chars and digits, length s_len """
    return ''.join(np.random.choice(list(string.ascii_letters + string.digits),
                                    s_len))


def d1d2(s, vol, T, rf, K):
    """ d1, d2 for Black-Scholes"""
    sigT = vol * np.sqrt(T)
    d1 = (np.log(s / K) + (rf + 0.5 * vol ** 2) * T) / sigT
    d2 = d1 - sigT
    return d1, d2


def black_scholes(s, K, vol, rf, T):
    """ """
    if T <= 0:
        if (type(s) == np.ndarray):
            nz = len(s)
            cp = np.zeros((nz,))
            pp = np.zeros((nz,))
            ind = (s > K)
            cp[ind] = (s - K)[ind]
            pp[~ind] = (K - s)[~ind]
        else:
            cp = s - K if s > K else 0
            pp = K - s if s < K else 0
    else:
        d1, d2 = d1d2(s, vol, T, rf, K)
        cp = norm.cdf(d1) * s - norm.cdf(d2) * K * np.exp(-rf * T)
        pp = -norm.cdf(-d1) * s + norm.cdf(-d2) * K * np.exp(-rf * T)
    return cp, pp


if use_cyrillic:
    lower_pos = 'нижний'
    medium_pos = 'средний'
    upper_pos = 'верхний'
    put = 'пут'
    call = 'колл'
    long = 'длинная'
    short = 'короткая'
    snp500_xlabel = 'S&P 500'
    buc_payoff_plot_title = 'Платежная функция портфеля бабочка-колл'
    bup_payoff_plot_title = 'Платежная функция портфеля бабочка-пут'
    x_pay_lab = 'Цена базового актива S'
    y_pay_lab = 'Платеж'
    pay_tit = 'Платежные функции опционов'
    MC_chart_title = 'Траектории S&P 500, Монте-Карло'
    hist_chart_title = 'Траектории S&P 500, историческое моделирование'
    hist_MC_chart_xlabel = 'Рабочие дни'
    hist_MC_chart_ylabel = 'Значения S&P 500'
    ti_h = 'Последнее значение в историческом моделировании'
    ti_mc = 'Последнее значение в моделировании Монте-Карло'
    port_payoff_lab = 'nлатеж портфеля'
    port_payoff_tit = 'Эмпирическая ФР платежа портфеля'
    port_perc_tit = 'Квантили'
    delta_lab = 'Дельта'
    distr_hist_tit = 'Распределение на {0} (история)'
    distr_mc_tit = 'Распределение на {0} (Монте-Карло)'
    simul_lab = 'Генерация'
    straddle_payoff = 'Платеж портфеля стрэдл'
    asset_price = 'Текущая цена базового актива'
    payoff = 'Платеж'
    bull_spread_payoff = 'Платеж бычьего спрэда'
    bear_spread_payoff = 'Платеж медвежьего спрэда'
    zero_level = 'Нулевой уровень'
    long_call = 'Длинный колл'
    short_call = 'Короткий колл'
else:
    lower_pos = 'lower'
    medium_pos = 'medium'
    upper_pos = 'upper'
    put = 'put'
    call = 'call'
    long = 'long'
    short = 'short'
    snp500_xlabel = 'S&P 500 value'
    buc_payoff_plot_title = 'Payoff of call butterfly portfolio'
    bup_payoff_plot_title = 'Payoff of a put butterfly portfolio'
    x_pay_lab = 'Underlying price S'
    y_pay_lab = 'Payoff'
    pay_tit = 'Options payoffs'
    MC_chart_title = 'Monte Carlo simulated S&P 500 paths'
    hist_chart_title = 'Historic simulated S&P 500 paths'
    hist_MC_chart_xlabel = 'Business days'
    hist_MC_chart_ylabel = 'S&P 500 values'
    ti_h = 'Distribution of historic simulation final value'
    ti_mc = 'Distribution of Monte Carlo simulation final value'
    port_payoff_lab = 'Portfolio payoff'
    port_payoff_tit = 'Empiric portfolio payoff CDF'
    port_perc_tit = 'Percentiles'
    delta_lab = 'Delta'
    distr_hist_tit = 'Distribution at {0} (historic)'
    distr_mc_tit = 'Distribution at {0} (Monte Carlo)'
    simul_lab = 'Simulation'
    straddle_payoff = 'Payoff of the straddle portfolio'
    asset_price = 'asset_price'
    payoff = 'Payoff'
    bull_spread_payoff = 'Payoff of the bull spread portfolio'
    bear_spread_payoff = 'Payoff of the bear spread portfolio'
    zero_level = 'Zero level'
    long_call = 'Long call'
    short_call = 'Short call'


def make_label_dict():
    """ create cyrillic or english labels """
    dd = {'put': 'пут', 'call': 'колл', 'lower': 'нижний', 'medium': 'средний',
          'upper': 'верхний', 'long': 'длинная', 'short': 'короткая'}
    if not use_cyrillic:
        dd = {li: li for li in list(dd.keys())}
    return dd


class Option:
    """ European option """

    def __init__(self, strike=50, call_op=True,
                 expires='2017-12-01', volume=1, name=''):
        """ """
        self.volume = volume
        self.K = strike
        self.call_option = call_op
        self.expires = pd.to_datetime(expires)
        self.name = name if len(name) > 0 else random_string(10)

    def time_to_maturity(self, cur_dat):
        """ """
        return np.array((self.expires - pd.to_datetime(cur_dat)).days / 365)

    def calc_price(self, s, vol, rf, dat):
        """ calculate price at date 'dat' with current undelying value 's',
        time to maturity 'T', risk-free rate 'rf'
        """
        T = self.time_to_maturity(dat)
        cp, pp = black_scholes(s, self.K, vol, rf, T)
        pr = cp if self.call_option else pp
        return pr * self.volume

    def greek_delta(self, s, vol, rf, cur_dat):
        """ option delta """
        T = self.time_to_maturity(cur_dat)
        d1, d2 = d1d2(s, vol, T, rf, self.K)
        if self.call_option:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def payoff(self, cur_s, s, vol, rf, cur_dat, dat=''):
        """ payoff from a future date 'dat' given future underlying value 's',
        current underlying value s_cur (at 'cur_dat'), and market vol, rf
        """
        if len(dat) == 0:
            dat = self.expires
        dd = pd.to_datetime(dat) - pd.to_datetime(cur_dat)
        dat_dif = np.array(dd.days / 365)
        future_price = self.calc_price(s, vol, rf, dat)
        current_price = self.calc_price(cur_s, vol, rf, cur_dat)
        po = future_price * np.exp(-rf * dat_dif) - current_price
        po = pd.Series(po, index=s)
        po.index.name = 'S'
        po.name = 'Payoff'
        return po

    def legend_label(self):
        """ plot labels, cyrillic/english """
        if use_cyrillic:
            lab_call = 'колл' if self.call_option else 'пут'
            lab_long = 'длинная' if self.volume > 0 else 'короткая'
        else:
            lab_call = 'call' if self.call_option else 'put'
            lab_long = 'long' if self.volume > 0 else 'short'
        return '_'.join([lab_call, lab_long])


class Portfolio(UserList):
    """ portfolio of options on the same underlying
    """

    def calc_prices(self, s, vol, rf, dat):
        """ """
        pr = np.array([op.calc_price(s, vol, rf, dat) for op in self.data])
        return pr

    def calc_price(self, s, vol, rf, dat):
        """ """
        return self.calc_prices(s, vol, rf, dat).sum()

    def payoff(self, cur_s, s, vol, rf, cur_dat, dat):
        """ payoff from a future date 'dat' given future underlying value 's',
        current underlying value cur_s (at 'cur_dat'), and market vol, rf
        """
        # ss = pd.Series(s)
        # ss = np.array(pd.Series(s))
        ss = pd.Series(s)
        dfo = pd.DataFrame()
        for op in self.data:
            opp = op.payoff(cur_s, np.array(ss), vol, rf, cur_dat, dat)
            dfo[op.name] = opp
        po = dfo.sum(axis=1)
        po.index = ss
        return po


def butterfly(Kl, Km, Ku, call_ops=True, expir='2017-12-01', vlm=1):
    """ long options with Kl, Ku strikes, two short options with Km strike,
    Kl < Km < Ku; inverse (short) Butterfy if volume vlm < 0
    """
    bu = Portfolio([Option(strike=Kl, call_op=call_ops,
                           expires=expir, volume=vlm, name=lower_pos),
                    Option(strike=Ku, call_op=call_ops,
                           expires=expir, volume=vlm, name=upper_pos),
                    Option(strike=Km, call_op=call_ops,
                           expires=expir, volume=-2 * vlm, name=medium_pos)])
    return bu


def straddle(K, expir='2017-12-01', vlm=1):
    """ """
    bu = Portfolio([Option(strike=K, call_op=True,
                           expires=expir, volume=vlm, name=lower_pos),
                    Option(strike=K, call_op=False,
                           expires=expir, volume=vlm, name=upper_pos)])
    return bu


def bull_spread(Kl, Ks, call_ops=True, expir='2017-12-01', vlm=1):
    """ """
    bu = Portfolio([Option(strike=Kl, call_op=call_ops,
                           expires=expir, volume=vlm, name=long_call),
                    Option(strike=Ks, call_op=call_ops,
                           expires=expir, volume=-vlm, name=short_call)])
    return bu

# ================================================================
# S&P 500 stuff
# ================================================================


class SnP500_data():
    """ """

    def __init__(self):
        """ """
        df = pd.read_csv('data/snp_adj_close.csv').set_index('date')['1972':]
        df['ret'] = df.adj_close.pct_change()
        df['mon'] = [s[:7] for s in df.index.astype(str)]
        df['year'] = [s[:4] for s in df.index.astype(str)]
        self.df = df

    def annual_vols(self):
        """ annualized volatilities computed by yearly data """
        gby = self.df.groupby('year')
        lst = [dfs.ret.std() * np.sqrt(dfs.shape[0])
               for key, dfs in gby]
        ks = [key for key, dfs in gby]
        dfv = pd.DataFrame(lst, columns=['volatility'], index=ks)
        dfv.drop(dfv.index[-1], inplace=True)
        return dfv

    def make_hist_returns(self, start='2012-01-01', finish='2025-12-31',
                          n_paths=100, n_days=75):
        """ get hisyoric returns from start to end,
        generate 'n_paths' paths for 'n_days' days
        """
        ret = self.df[(self.df.index >= start) & (self.df.index <= finish)].ret
        rets = np.concatenate((np.ones((1, n_paths)),
                               np.random.choice(1 + ret,
                                                size=(n_days, n_paths))))
        return rets.cumprod(axis=0)

    def make_mc_returns(self, start='2012-01-01', finish='2025-12-31',
                        n_paths=100, n_days=75):
        """ """
        ret = self.df[(self.df.index >= start) & (self.df.index <= finish)].ret
        ret_daily = ret.mean()
        sigma_daily = ret.std()
        ret_mc = np.random.normal(loc=ret_daily, scale=sigma_daily,
                                  size=(n_days, n_paths))
        rets = np.concatenate((np.ones((1, n_paths)), 1 + ret_mc))
        return rets.cumprod(axis=0)

    def last_value(self):
        """ """
        return self.df.adj_close[self.df.index[-1]]


def ecdf(se):
    """ """
    cdf = se.sort_values()
    n = len(se)
    ser = pd.Series(np.ones((n,)) / n, index=cdf)
    return ser.cumsum()
