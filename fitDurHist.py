# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:27:25 2015

@author: pavel
"""
from __future__ import division
from matplotlib.pylab import *
from numpy import *
import scipy.stats
import scipy.optimize

T = 20.3

(TimesUp, TimesDown, Periods) = np.load("durat_182.5_1.3_5.26_2.1_.npy")
TimesUp = np.array(TimesUp, dtype='float32')
TimesDown = np.array(TimesDown, dtype='float32')

TimesUp *= T
TimesDown *= T

maxHst = 500000
minHst = 500
binSz = 500
D = 15000

hst, bins = histogram(TimesUp, bins=(maxHst-minHst)/binSz, range=(minHst, maxHst))
bins = bins[:-1]
#%%
def expFun(x, *args):
    A, scale = args
    return A*exp(-scale*(x - minHst))

def powFun(x, *args):
    A, scale = args
    return A/x**scale

def mixedFun(x, *args):
    A1, A2, scale1, scale2, scale3, b = args
    a = 1/(1 + exp(-scale3*(x - b)))
    return (1 - a)*A1/x**scale1 + a*A2*exp(-scale2*x)
#%%
# left hand of distribution (power law)
bins1 = bins[:D/binSz]
hst1 = hst[:D/binSz]
params = [1, 0.01]
fitted_params,_ = scipy.optimize.curve_fit(powFun, bins1, hst1, p0=params)

figure(1)
plot(bins1, hst1)
plot(bins1, powFun(bins1, *fitted_params), 'r')
xlabel("Duration[ms]")
title("Power law fitting")
legend(["data", "fitted"])
#%%
# right hand of distribution (exponential)
bins2 = bins[D/binSz:]
hst2 = hst[D/binSz:]
#params = [10, 0.0001]
params = [60, 1.2e-05]
fitted_params,_ = scipy.optimize.curve_fit(expFun, bins2, hst2, p0=params)

figure(2)
plot(bins2, hst2)
plot(bins2, expFun(bins2, *fitted_params), 'r')
xlabel("Duration[ms]")
title("Exponential")
legend(["data", "fitted"])
#%%
params = [3.05709573e+08, 1960., 1.76, 2.3e-05, 0.0025, 5000]
fitted_params,_ = scipy.optimize.curve_fit(mixedFun, bins, hst, p0=params)
A1, A2, scale1, scale2, scale3, b = fitted_params
#%%
figure(3)
loglog(bins, hst, '.')
loglog(bins, mixedFun(bins, *fitted_params), '--r', linewidth=1.5)
xlabel(r"$Duration\ Times\ Up,\ ms$")
ylabel(r"$Counts$")
title("Mixed fitting")
legend(["data", "fitted"])
#%%
hst, bins = histogram(TimesDown, bins=(maxHst-minHst)/binSz, range=(minHst, maxHst))
bins = bins[:-1]

params = [3.46+04, 60., 0.753, 1.2e-05, 0.25, 1000]
fitted_params,_ = scipy.optimize.curve_fit(mixedFun, bins, hst, p0=params)
A1, A2, scale1, scale2, scale3, b = fitted_params

figure(4)
loglog(bins, hst, '.')
loglog(bins, mixedFun(bins, *fitted_params), '--r', linewidth=1.5)
xlabel(r"$Duration\ Times\ Down,\ ms$")
ylabel(r"$Counts$")
title("Mixed fitting")
legend(["data", "fitted"])
