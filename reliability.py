import re
from sys import maxsize as sys_maxsize

from datetime import datetime as dt_datetime

import numpy as np

from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy.stats import norm

import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from pyspark.sql import functions as ps_func
from pyspark.sql.functions import pandas_udf
from pyspark.sql.functions import PandasUDFType
from pyspark.sql.types import DoubleType
from pyspark.sql.types import StructField
from pyspark.sql.window import Window
from pyspark.sql import types as ps_types

def plot_fit(x_data, y_data, x_pred, y_pred):
    plt.plot(x_data, y_data, 'o', label='original data')
    plt.plot(x_pred, y_pred, 'r', label='fitted line')
    plt.legend()
    plt.show()

def check_fit(y_data, y_pred, w_data):

    if w_data is None:
        sigma = w_data
    else:
        sigma = np.sqrt(w_data)

    mse = metrics.mean_squared_error(y_data, y_pred, sigma)
    mae = metrics.mean_absolute_error(y_data, y_pred, sigma)
    rsq = metrics.r2_score(y_data, y_pred, sigma)

    return mse, mae, rsq

def sf_weibull(x, s, loc, scale):
    return stats.weibull_min.sf(x, s, loc=loc, scale=scale)

def sf_weibull_noshift(x, s, scale):
    return stats.weibull_min.sf(x, s, loc=0, scale=scale)

def estimate_weibull(x_data, y_data, w_data=None, shift=False, wlimit=24, maxfev=1000, ftol=1e-10, xtol=1e-10):

    x_data_log = x_data[~np.equal(y_data, np.ones(y_data.shape[0]))]
    y_data_log = y_data[~np.equal(y_data, np.ones(y_data.shape[0]))]

    log_x = np.log(x_data_log)
    log_mlog_y = np.log(-np.log(y_data_log))

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_mlog_y)

    # derive weibull parameters
    k = slope
    s = np.exp(-(intercept/slope))
    inital = np.array([k, 0, s])

    if w_data is None:
        sigma = w_data
    else:
        sigma = 1/np.sqrt(w_data)

    if not shift:
        real_inital = np.array([k, s])
        sf_weibull_used = sf_weibull_noshift
    else:
        real_inital = inital
        sf_weibull_used = sf_weibull

    out = curve_fit(sf_weibull_used,
                    x_data,
                    y_data,
                    sigma=sigma,
                    method='lm',
                    p0=real_inital,
                    full_output=True,
                    maxfev=maxfev,
                    ftol=ftol,
                    xtol=xtol)

    popt = out[0]

    if not shift:
        popt = np.array([out[0][0], 0, out[0][1]])

    infodict = out[2]['nfev']
    mesg = re.sub(' +', ' ', out[3].replace('\n', '').replace('\n', ''))
    ier = out[4]

    y_pred = sf_weibull(x_data, *popt)

    mse, mae, rsq = check_fit(y_data, y_pred, w_data)

    print("--- Weibull assumption")
    print("-- Linearized problem")
    print("r square: %f\np value: %f\nstandard error: %f" % (r_value, p_value, std_err))
    print("shape: %f, loc: %f scale: %f" % (inital[0], inital[1], inital[2]))
    print("-- Nonlinear problem")
    print("shape: %f, loc: %f scale: %f" % (popt[0], popt[1], popt[2]))
    print("mean square error: %f\nmean absolute error: %f\nr square: %f" % (mse, mae, rsq))
    print("reliablilty: %f" % sf_weibull(wlimit, *popt))

    x_sample_weibull = np.linspace(0.1, wlimit, num=100)
    y_pred_weibull = sf_weibull(x_sample_weibull, *popt)

    plot_fit(log_x, log_mlog_y, log_x, intercept + slope*log_x)
    plot_fit(x_data, y_data, x_data, y_pred)
    plot_fit(x_data, y_data, x_sample_weibull, y_pred_weibull)

    return inital[0], inital[2], popt[0], popt[1], popt[2], mse, mae, rsq, infodict, mesg, ier

def sf_lognormal(x, s, loc, scale):
    return stats.lognorm.sf(x, s, loc, scale=scale)

def sf_lognormal_noshift(x, s, scale):
    return stats.lognorm.sf(x, s, loc=0, scale=scale)

def estimate_lognormal(x_data, y_data, w_data=None, shift=False, wlimit=24, maxfev=1000, ftol=1e-10, xtol=1e-10):
    x_data_log = x_data[~np.equal(y_data, np.ones(y_data.shape[0]))]
    y_data_log = y_data[~np.equal(y_data, np.ones(y_data.shape[0]))]

    log_x = np.log(x_data_log)
    inv_y = norm.ppf(1 - y_data_log, loc=0, scale=1)

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, inv_y)

    # derive normal parameters
    k = 1/slope
    s = -intercept/slope

    # A common parametrization for a lognormal random variable is in terms of the mu and sigma
    # of the unique normally distributed random variable
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html#scipy.stats.lognorm
    scale = np.exp(s)
    s = k

    inital = np.array([s, 0, scale])

    if w_data is None:
        sigma = np.ones(y_data.shape[0])
    else:
        sigma = 1/np.sqrt(w_data)

    if not shift:
        real_inital = np.array([s, scale])
        sf_lognormal_used = sf_lognormal_noshift
    else:
        real_inital = inital
        sf_lognormal_used = sf_lognormal

    out = curve_fit(sf_lognormal_used,
                    x_data,
                    y_data,
                    sigma=sigma,
                    method='lm',
                    p0=real_inital,
                    full_output=True,
                    maxfev=maxfev,
                    ftol=ftol,
                    xtol=xtol)

    popt = out[0]

    if not shift:
        popt = np.array([out[0][0], 0, out[0][1]])

    infodict = out[2]['nfev']
    mesg = re.sub(' +', ' ', out[3].replace('\n', '').replace('\n', ''))
    ier = out[4]

    y_pred = sf_lognormal(x_data, *popt)

    mse, mae, rsq = check_fit(y_data, y_pred, w_data)

    print("--- LogNormal assumption")
    print("-- Linearized problem")
    print("r square: %f\np value: %f\nstandard error: %f" % (r_value, p_value, std_err))
    print("location: %f, loc: %f scale: %f" % (inital[0], inital[1], inital[2]))
    print("-- Nonlinear problem")
    print("location: %f, loc: %f scale: %f" % (popt[0], popt[1], popt[2]))
    print("mean square error: %f\nmean absolute error: %f\nr square: %f" % (mse, mae, rsq))
    print("reliablilty: %f" % sf_lognormal(wlimit, *popt))

    x_sample_lognormal = np.linspace(0.1, wlimit, num=100)
    y_pred_lognormal = sf_lognormal(x_sample_lognormal, *popt)

    plot_fit(log_x, inv_y, log_x, intercept + slope*log_x)
    plot_fit(x_data, y_data, x_data, y_pred)
    plot_fit(x_data, y_data, x_sample_lognormal, y_pred_lognormal)

    return inital[0], inital[2], popt[0], popt[1], popt[2], mse, mae, rsq, infodict, mesg, ier