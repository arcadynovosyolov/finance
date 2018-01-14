
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import time as tt


def f_plus(x):
    """ 'plus' function """
    return np.array((x > 0), dtype=int) * x


def preliminaries():
    """ to run by F9 """
    mark = mark_def()
    stock = stock_def(S0=100, sigma=0.2, cnt=1)
    opt_call = option_def(K=100, call=True, T=1, cnt=1)
    opt_put = option_def(K=100, call=False, T=1, cnt=1)
    paths = stock_paths(stock)
    deltas_call = option_deltas(opt_call, stock, mark)
    deltas_put = option_deltas(opt_put, stock, mark)
    return paths, deltas_call, deltas_put


def mark_def(r=0.03):
    """ """
    mark = {'r': r}
    return mark


def stock_def(S=100, sigma=0.2, cnt=1):
    """ """
    stock = {'S': S, 'sigma': sigma, 'cnt': cnt}
    return stock


def option_def(K=100, call=True, T=1, cnt=1):
    """ options position
    K - strike price
    call = True/False
    T - expires in T years
    cnt - count of options if a position (negative for short position)
    """
    return {'K': K, 'call': call, 'T': T, 'cnt': cnt}


def portfolio_distrib(opts, mark, stock, horizon, n_trials):
    """ """
    pv = portfolio_value(opts, mark, stock)
    n_days = 250
    c_opts = list()
    dT = horizon / n_days
    for x in opts:
        c_opts.append(option_def(x['K'], x['call'], x['T'] - dT, x['cnt']))
    hor_vol = stock['sigma'] * np.sqrt(dT)

    # calculate stock values at the horizon end
    rets = np.random.normal(scale=hor_vol, size=(n_trials,))
    # vals = stock['S'] * np.exp(rets)    # exponential returns
    vals = stock['S'] * (1 + rets)      # arithmetic returns

    pvals = np.zeros((n_trials,))
    for i in range(n_trials):
        pvals[i] = portfolio_value(c_opts, mark,
                                   stock_def(vals[i],
                                             stock['sigma'],
                                             stock['cnt']))
    return pvals - pv


def portfolio_value(opts, mark, stock):
    """ portfolio contents
    opts - list of options (negative cnt for short position)
    stock - underlying stock (negative cnt for short position,
            zero cnt for options-only portfolio)
    mark - market definition (r)
    """
    vls = [black_scholes(x, mark, stock) for x in opts]
    current_value = sum(vls) + stock['S'] * stock['cnt']
    return current_value


def d1d2(option, mark, stock):
    sigT = stock['sigma'] * np.sqrt(option['T'])
    d1 = (np.log(stock['S'] / option['K']) + (mark['r'] +
          0.5 * stock['sigma'] ** 2) * option['T']) / sigT
    d2 = d1 - sigT
    return d1, d2


def stock_paths(stock, n_paths=5, n_days=250, T=1, show_plots=True):
    """
    paths for T years duration
    stock - stock obkect (current price, volatility)
    n_paths - number of sample paths
    n_days - number of business days in a year
    """
    sig = stock['sigma'] / np.sqrt(n_days)  # daily volatility
    np.random.seed(0)                       # for reproducibility
    sz = int(T * n_days + 1)                # number of days
    rets = np.random.normal(0, sig, (sz, n_paths))
    paths = np.ones((sz, n_paths)) * stock['S']
    for i in range(1, sz):
        paths[i, :] = paths[i - 1, :] * np.exp(rets[i, :])
    if show_plots:
        plot_paths(paths)
    return paths


def option_deltas(option, stock, mark, n_days=250, show_plots=True):
    """ """
    paths = stock_paths(stock, n_days=n_days, T=option['T'], show_plots=False)
    times_to_expir = option['T'] * np.ones(paths.shape)
    n = paths.shape[0] - 1
    m = paths.shape[1]
    times_to_expir = option['T'] * (1 - np.array(range(n + 1)) / n)
    deltas = np.ones((n, m))
    for i in range(n):
        for j in range(m):
            deltas[i, j] = greek_delta(option_def(option['K'],
                                                  option['call'],
                                                  times_to_expir[i],
                                                  option['cnt']),
                                       mark,
                                       stock_def(paths[i, j],
                                                 stock['sigma'],
                                                 stock['cnt']))
    if show_plots:
        plot_deltas(deltas)
    return deltas


def option_values(option, stock, mark, n_days=250, show_plots=True):
    """ """
    paths = stock_paths(stock, n_days=n_days, T=option['T'], show_plots=False)
    times_to_expir = option['T'] * np.ones(paths.shape)
    n = paths.shape[0]
    m = paths.shape[1]
    times_to_expir = option['T'] * (1 - np.array(range(n + 1)) / n)
    mult = 1 if option['call'] else -1
    vals = np.ones((n, m))
    for i in range(n - 1):
        for j in range(m):
            vals[i, j] = black_scholes(option_def(option['K'],
                                                  option['call'],
                                                  times_to_expir[i],
                                                  option['cnt']),
                                       mark,
                                       stock_def(paths[i, j],
                                                 stock['sigma'],
                                                 stock['cnt']))
    vals[n-1, :] = f_plus(mult * (paths[n-1, :] - option['K']))
    if show_plots:
        plot_vals(vals, option['call'])
    return vals


def plot_vals(vals, call=True, tit='Sample values along the paths', fig_num=0):
    """ """
    # call = True if deltas[0, 0] > 0 else False
    for j in range(vals.shape[1]):
        plt.plot(vals[:, j])
    plt.ylabel('Value')
    plt.xlabel('Business days')
    tit_more = 'call' if call else 'put'
    plt.title('Values along sample paths, {0} option'.format(tit_more))
    plt.savefig('fig_{0}.png'.format(fig_num))


def plot_deltas(deltas, tit='Sample deltas along the paths', fig_num=0):
    """ """
    call = True if deltas[0, 0] > 0 else False
    for j in range(deltas.shape[1]):
        plt.plot(deltas[:, j])
    plt.ylabel('Delta')
    plt.xlabel('Business days')
    tit_more = 'call' if call else 'put'
    plt.title('Deltas along sample paths, {0} option'.format(tit_more))
    cl = ['delta_{0}'.format(i) for i in range(deltas.shape[1])]
    df = pd.DataFrame(deltas, columns=cl)
    df.to_csv('fig_{0}.csv'.format(fig_num), index=None)
    plt.savefig(fig_name(fig_num))
    #   plt.savefig('deltas_{0}.png'.format(tit_more))


def plot_paths(paths, tit='Sample paths', fig_num=0):
    """ """
    for j in range(paths.shape[1]):
        plt.plot(paths[:, j])
    plt.ylabel('Price, $')
    plt.xlabel('Business days')
    plt.title('Sample paths')
    cl = ['path_{0}'.format(i) for i in range(paths.shape[1])]
    df = pd.DataFrame(paths, columns=cl)
    df.to_csv('fig_{0}.csv'.format(fig_num), index=None)
    plt.savefig(fig_name(fig_num))


def fig_name(fig_num=0):
    """ """
    return 'images/fig_{0}.png'.format(fig_num)


def csv_let_name(let='A', fig_num=0):
    """ """
    return 'csv/fig_{0}_{1}.csv'.format(fig_num, let)


def csv_name(fig_num=0):
    """ """
    return 'csv/fig_{0}.csv'.format(fig_num)


def delta_hedged_position(option, mark, stock):
    """ """
    dl = greek_delta(option, mark, stock)
    opv = black_scholes(option, mark, stock)
    mlt = opv * dl
    option_cnt = - stock['S'] * stock['cnt'] / mlt
    return option_def(K=option['K'], T=option['T'],
                      call=option['call'], cnt=option_cnt)

call_names = ['put', 'call']


def portfolio_descr(opts, mark, stock):
    """ """
    df = pd.concat((stock_descr(stock),
                    options_descr(opts, mark, stock)),
                   axis=0)
    df.index = range(df.shape[0])
    return df


def stock_descr(stock):
    """ """
    df = pd.DataFrame({'cnt': stock['cnt'], 'delta': 1,
                       'hedging_pos': -1,
                       'value': stock['S'] * stock['cnt'],
                       },
                      index=[0])
    return df


def options_descr(opts, mark, stock):
    """ """
    st = True
    for x in opts:
        if st:
            df = option_descr(x, mark, stock)
            st = False
        else:
            dff = option_descr(x, mark, stock)
            df = pd.concat([df, dff], axis=0)
    df['call'] = df['call'].apply(lambda x: call_names[x])
    df.index = range(df.shape[0])
    return df


def option_descr(option, mark, stock):
    """ """
    df = pd.DataFrame(option, index=[0])
    df['delta'] = greek_delta(option, mark, stock)
    df['value'] = black_scholes(option, mark, stock)
    df['hedging_pos'] = delta_hedged_position(option, mark, stock)['cnt']
    return df


def greek_delta(option, mark, stock):
    """ option delta """
    d1, d2 = d1d2(option, mark, stock)
    if option['call']:
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def black_scholes(opt, mark, stock):
    """ Black-Scholes formula for plain vanilla options """
    d1, d2 = d1d2(opt, mark, stock)
    if opt['call']:
        pr = norm.cdf(d1) * stock['S'] - \
             norm.cdf(d2) * opt['K'] * np.exp(-mark['r'] * opt['T'])
    else:
        pr = -norm.cdf(-d1) * stock['S'] + \
             norm.cdf(-d2) * opt['K'] * np.exp(-mark['r'] * opt['T'])
    return pr * opt['cnt']

# -------------------------------------------------------------
# not used in the cirrent version
# -------------------------------------------------------------


def binomial_tree(opt, mark, stock, N=2000, american=False):
    """ binomial tree for american and european plain vanilla options
    N - tree height
    """

    # lattice parameters
    deltaT = float(opt['T']) / N
    u = np.exp(stock['sigma'] * np.sqrt(deltaT))
    d = 1.0 / u

    # working arrays
    fs = np.asarray([0.0 for i in range(N + 1)])
    fs2 = np.asarray([(stock['S'] * u**j * d**(N - j)) for j in range(N + 1)])
    fs3 = np.asarray([float(opt['K']) for i in range(N + 1)])

    # rates are fixed so the probability of up and down are fixed.
    # this is used to make sure the drift is the risk free rate
    a = np.exp(mark['r'] * deltaT)
    p = (a - d) / (u - d)
    oneMinusP = 1.0 - p

    # Compute the leaves, f_{N, j}
    if opt['call']:
        fs[:] = np.maximum(fs2 - fs3, 0.0)
    else:
        fs[:] = np.maximum(-fs2 + fs3, 0.0)

    # calculate backward the option prices
    for i in range(N - 1, -1, -1):
        fs[:-1] = np.exp(- mark['r'] * deltaT) * (p * fs[1:] +
                                                  oneMinusP * fs[:-1])
        fs2[:] = fs2[:] * u

        if american:
            # Simply check if the option is worth more alive or dead
            if opt['call']:
                fs[:] = np.maximum(fs[:], fs2[:] - fs3[:])
            else:
                fs[:] = np.maximum(fs[:], -fs2[:] + fs3[:])

    return fs[0]


def stocks_dict_1():
    """ example list consusting of a single stock """
    dct = {'First': stock_def(100, 0.2, 1)}
    return dct


def stocks_dict_2():
    """ example list consusting of a single stock """
    dct = {'First': stock_def(100, 0.2, 1)}
    dct['Second'] = stock_def(70, 0.15, 2)
    return dct


def def_inderlyings():
    """ """
    return {'First': stock_def(100, 0.2), 'Second': stock_def(70, 0.15)}


# in case one needs it
def time_info(start_time):
    """ """
    return '{0:10.2f}'.format(tt.time() - start_time).strip()
