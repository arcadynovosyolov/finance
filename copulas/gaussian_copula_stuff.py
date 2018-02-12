# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 14:16:36 2017

@author: Arcady Novosyolov
"""

import numpy as np
import pandas as pd
import scipy.stats as ss
from statsmodels.sandbox.distributions.extras import mvnormcdf
import matplotlib.pyplot as plt

# avoiding boundary trouble
loc_eps = 0.000001

# ================================================================
# multivariate distribution with normal copula
# ================================================================


class MarginalDistribution:
    """ """

    def set_params(self, marg_typ='normal', params=(0, 1)):
        """ """
        self.typ = marg_typ
        self.params = params

    def fit(self, data, marg_typ='normal'):
        """ """
        if marg_typ == 't':
            params = ss.t.fit(data)
        elif marg_typ == 'expon':
            params = ss.expon.fit(data)
        else:
            params = ss.norm.fit(data)
        self.set_params(marg_typ, params)

    def pdf(self, x):
        """ """
        pr = self.params
        if self.typ == 't':
            y = ss.t.pdf(x, *pr)
        elif self.typ == 'expon':
            y = ss.expon.pdf(x, *pr)
        else:
            y = ss.norm.pdf(x, *pr)
        return y

    def cdf(self, x):
        """ """
        pr = self.params
        if self.typ == 't':
            y = ss.t.cdf(x, *pr)
        elif self.typ == 'expon':
            y = ss.expon.cdf(x, *pr)
        else:
            y = ss.norm.cdf(x, *pr)
        return y

    def ppf(self, x):
        """ """
        pr = self.params
        if self.typ == 't':
            y = ss.t.ppf(x, *pr)
        elif self.typ == 'expon':
            y = ss.expon.ppf(x, *pr)
        else:
            y = ss.norm.ppf(x, *pr)
        return y


class MultiVarDistribution:
    """ """

    def fit(self, data, marg_typ='t'):
        """ """
        data = pd.DataFrame(data).interpolate().ffill().bfill()
        cr = pearson_from_spearman(data.corr(method='spearman'))
        self.data = data
        self.set_corr(cr)
        cls = data.columns
        margs = pd.Series()
        for cl in cls:
            mrg = MarginalDistribution()
            mrg.fit(data[cl], marg_typ)
            margs[cl] = mrg
        self.set_marginals(margs)

    def set_corr(self, cr):
        """ """
        self.cop = GaussianCopula(corr_matr=cr)

    def set_marginals(self, marginals):
        """ sets the predefined Series of marginal parameters """
        self.marginals = marginals

    def marginal_pdf(self, x, cl):
        """ univariate marginal PDF """
        return self.marginals[cl].pdf(x)

    def marginal_cdf(self, x, cl):
        """ univariate marginal CDF """
        return self.marginals[cl].cdf(x)

    def marginal_ppf(self, x, cl):
        """ univariate marginal PPF """
        return self.marginals[cl].ppf(x)

    def u_from_x(self, x):
        """ apply marginal CDF to each component """
        u = pd.DataFrame(index=x.index, columns=x.columns)
        for cl in x.columns:
            u[cl] = self.marginal_cdf(x[cl], cl)
        return u

    def marg_pdf(self, x):
        """ multivariate marginal probability density function """
        # independent target density
        id = pd.DataFrame(index=x.index, columns=x.columns)
        for cl in x.columns:
            id[cl] = self.marginals[cl].pdf(x[cl])
        i_dens = id.prod(axis=1)

        # compute U components
        u = self.u_from_x(x)

        # compute copula marginal density
        c_pdf = pd.Series(np.array(self.cop.marg_pdf(u)),
                          index=x.index)

        se = c_pdf * i_dens
        trg = ', '.join(x.columns)
        se.name = 'Marginal PDF of "{0}"'.format(trg)
        return se

    def marg_cdf(self, x):
        """ multivariate marginal cumulative distribution function """
        # compute U components
        u = self.u_from_x(x)

        # compute copula marginal CDF
        se = pd.Series(np.array(self.cop.marg_cdf(u)), index=x.index)
        trg = ', '.join(self.cop.targets)
        se.name = 'Marginal CDF of "{0}"'.format(trg)
        return se

    def cond_pdf(self, x, x_cond):
        """ conditional probability density function """
        # independent target density
        id = pd.DataFrame(index=x.index, columns=x.columns)
        for cl in x.columns:
            id[cl] = self.marginals[cl].pdf(x[cl])
        i_dens = id.prod(axis=1)

        # compute U components
        u = self.u_from_x(x)
        u_cond = self.u_from_x(x_cond)

        # compute copula conditional density
        c_pdf = pd.Series(np.array(self.cop.cond_pdf(u, u_cond)),
                          index=x.index)

        se = c_pdf * i_dens
        trg = ', '.join(self.cop.targets)
        cnd = ', '.join(self.cop.conditionals)
        se.name = 'Conditional PDF of "{0}" given "{1}"'.format(trg, cnd)
        return se

    def cond_cdf(self, x, x_cond):
        """ conditional cumulative distribution function """
        # compute U components
        u = self.u_from_x(x)
        u_cond = self.u_from_x(x_cond)

        # compute copula conditional CDF
        se = pd.Series(np.array(self.cop.cond_cdf(u, u_cond)), index=x.index)
        trg = ', '.join(self.cop.targets)
        cnd = ', '.join(self.cop.conditionals)
        se.name = 'Conditional CDF of "{0}" given "{1}"'.format(trg, cnd)
        return se

# ================================================================
# copula classes
# ================================================================


class GaussianCopula:
    """ """

    def __init__(self, corr_matr=pd.DataFrame(np.eye(2))):
        """
        initialize copula (as two dimensional independent one by default)
        """
        self.set_corr(corr_matr)

    def set_corr(self, cr):
        """ set correlation matrix parameter and compute its inverse """
        self.cr = pd.DataFrame(cr)
        self.dim = self.cr.shape[0]
        cls = self.cr.columns
        self.cri = pd.DataFrame(np.linalg.inv(self.cr), columns=cls, index=cls)

    def fit(self, data, alpha=None):
        """
        fit the copula from data
        the data should be 'm times n' Numpy array or Pandas DataFrame,
        m = number of samples (observations), n = number of components
        alpha for exponential weighting, about 0.01 -- 0.05
        """
        if alpha is None:
            cr = pd.DataFrame(data).corr(method='spearman')
        else:
            crt = pd.DataFrame(data).ewm(alpha=alpha).corr(method='spearman')
            cr = crt.loc[data.index[-1]]
        self.set_corr(pearson_from_spearman(cr))

    def rvs(self, size=100000):
        """
        simulate sample for Monte Carlo method
        size = sample volume, 100,000 by default
        """
        sam = ss.multivariate_normal.rvs(cov=self.cr, size=size)
        return pd.DataFrame(ss.norm.cdf(sam), columns=self.cr.columns)

    def pdf(self, u):
        """ probability density function of the copula """
        x = ss.norm.ppf(u)
        indep = ss.norm.pdf(x).prod(axis=1)
        dep = ss.multivariate_normal.pdf(x, cov=self.cr)
        return dep / indep

    def marg_pdf(self, u):
        """ marinal probability density function of the copula """
        targets = list(u.columns)
        marg_cr = self.cr.loc[targets, targets]
        marg_cri = pd.DataFrame(np.linalg.inv(marg_cr), index=targets,
                                columns=targets)
        x = ss.norm.ppf(u)
        mat = marg_cri - np.eye(len(targets))
        pok = -0.5 * x.dot(mat).dot(x.T)
        mult = 1 / np.sqrt(np.linalg.det(marg_cri))
        return pd.Series(mult * np.diag(np.exp(pok)), index=range(u.shape[0]))

    def marg_cdf(self, u):
        """
        u is DataFrame n_obs x targets
        """
        targets = list(u.columns)
        x = self.make_input(u)
        if len(self.targets) <= 1:
            # standard univariate normal
            res = ss.norm.cdf(x)
            res = pd.Series(res[:, 0], index=u.iloc[:, 0])
            #  res = pd.Series(res, index=u[:, 0])
        else:
            # mvnormcdf does not accept multiple input points
            res = np.zeros(x.shape[0])
            ml = [0] * len(targets)
            for i in range(x.shape[0]):
                xl = np.array(x.iloc[i, :])
                cv = np.array(self.cr.loc[targets, targets])
                res[i] = mvnormcdf(xl, ml, cv)
            res = pd.Series(res, index=range(u.shape[0]))
        res.name = 'Cond CDF of ' + ', '.join(self.targets)
        return res

    def probability(self, event_func, n_obs=100000):
        """
        calculates probability of an event A, which is defined via its
        indicator function "event_func"
        """
        u = self.rvs(size=n_obs)
        return event_func(u).mean()

    # --------------------------------------------------------------------
    # Regular conditional stuff
    # --------------------------------------------------------------------

    def cond_pdf(self, u, u_cond):
        """
        u is DataFrame n_obs x targets
        u_cond is DataFrame 1 x conditionals
        """
        self.fit_cond(targets=u.columns, conditionals=u_cond.columns)
        x, x_cond, mn = self.make_input(u, u_cond)
        if len(self.targets) <= 1:
            res = ss.norm.pdf(x, mn.iloc[0], self.cond_cov.iloc[0, 0] ** 0.5)
            fx = ss.norm.pdf(x)
            res = pd.Series((res / fx)[:, 0], index=u.iloc[:, 0])
        else:
            res = ss.multivariate_normal.pdf(x, mean=np.array(mn)[0],
                                             cov=self.cond_cov)
            fx = ss.norm.pdf(x).prod(axis=1)
            res = pd.Series(res / fx, index=range(x.shape[0]))
        res.name = 'Cond PDF of ' + ', '.join(self.targets)
        return res

    def cond_cdf(self, u, u_cond):
        """
        u is DataFrame n_obs x targets
        u_cond is DataFrame 1 x conditionals
        """
        self.fit_cond(targets=u.columns, conditionals=u_cond.columns)
        x, x_cond, mn = self.make_input(u, u_cond)
        if len(self.targets) <= 1:
            # univariate normal
            res = ss.norm.cdf(x, mn.iloc[0], self.cond_cov.iloc[0, 0] ** 0.5)
            res = pd.Series(res[:, 0], index=u.iloc[:, 0])
            #  res = pd.Series(res, index=u[:, 0])
        else:
            # mvnormcdf does not accept multiple input points
            res = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                xl = np.array(x.iloc[i, :])
                ml = np.array(mn)[0]
                cv = np.array(self.cond_cov)
                res[i] = mvnormcdf(xl, ml, cv)
            res = pd.Series(res, index=range(u.shape[0]))
        res.name = 'Cond CDF of ' + ', '.join(self.targets)
        return res

    def fit_cond(self, targets=None, conditionals=None):
        """
        conditinal variables list;
        target variables will be kept in 'targets' list
        """
        cls = self.cr.columns

        # by default conditionals contains everything except the last component
        if (conditionals is None) or (targets is None):
            conditionals = cls[:-1]
            targets = [cls[-1]]

        # keep the variables lists
        self.targets = targets
        self.conditionals = conditionals

        # covariance submatrices
        sII = np.array(self.cr.loc[conditionals, conditionals])
        sIJ = np.array(self.cr.loc[conditionals, targets])
        sJI = np.array(self.cr.loc[targets, conditionals])
        sJJ = np.array(self.cr.loc[targets, targets])

        # conditional mean and covariance
        pr = np.linalg.solve(sII, sIJ)
        self.cond_cov = pd.DataFrame(sJJ - sJI.dot(pr), index=targets,
                                     columns=targets)
        self.mupr = pd.DataFrame(pr.T, index=targets, columns=conditionals)

    def make_input(self, u, u_cond=None):
        """
        data preparation for conditional functions
        u and u_cond are DataFrames, Series, or arrays !!!
        """
        if u.ndim == 1:
            u = pd.DataFrame(u).T
        if (u_cond is not None):
            if (u_cond.ndim == 1):
                u_cond = pd.DataFrame(u_cond).T
        correct_u(u)
        x = pd.DataFrame(ss.norm.ppf(u), index=u.index, columns=u.columns)
        if u_cond is None:
            return x
        else:
            x_cond = pd.DataFrame(ss.norm.ppf(u_cond), index=u_cond.index,
                                  columns=u_cond.columns)
            mn = self.mupr.dot(x_cond.T).T
            return x, x_cond, mn

# =================================================================
# auxiliary
# =================================================================


def correct_u(u):
    """ remove boundary effects at 0 and 1 """
    u[u == 0] = loc_eps
    u[u == 1] = 1 - loc_eps

def spearman_from_pearson(r):
    """ convertion for normal multivariate """
    s = 6 / np.pi * np.arcsin(0.5 * r)
    return s


def pearson_from_spearman(s):
    """ conversion for normal multivariate """
    r = 2 * np.sin(s * np.pi / 6)
    return r
