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

T = 20.

(TimesUp, TimesDown, Periods) = np.load("durat_182.5_1.3_5.26_2.1.npy")
TimesUp *= T
TimesDown *= T
#%%
maxHst = 5000
binSz = 40

figure(1)
hist(TimesUp, bins=maxHst/binSz, range=(T, maxHst), edgecolor='none')
xlabel("Duration, ms")
title("TumesUp")

maxHst = 500000
minHst = 5000
binSz = 1600
figure(2)
hist(TimesUp, bins=(maxHst-minHst)/binSz, range=(minHst, maxHst), edgecolor='none')
xlabel("Duration, ms")
title("TumesUp")
#%%
def expFun(x, *args):
    A, scale = args
    return A*exp(-scale*(x - minHst))

def powFun(x, *args):
    A, scale = args
    return A/x**scale

maxHst = 500000
#minHst = 5000
#maxHst = 5000
minHst = 80
binSz = 80

hst, bins = histogram(TimesUp, bins=(maxHst-minHst)/binSz, range=(minHst, maxHst))
bins = bins[:-1]
#%%
params = [1, 0.01]
fitted_params,_ = scipy.optimize.curve_fit(powFun, bins, hst, p0=params)

figure(1)
plot(bins, hst)
plot(bins, powFun(bins, *fitted_params), 'r')
#%%
def Fun(x, *args):
    A1, A2, scale1, scale2, scale3, b = args
    a = 1/(1 + exp(-scale3*(x - b)))
    return (1 - a)*A1/x**scale1 + a*A2*exp(-scale2*(x - b))

params = [3.05709573e+08, 196., 1.76, 2.3e-05, 0.0025, 5000]
fitted_params,_ = scipy.optimize.curve_fit(Fun, bins, hst, p0=params)
plot(bins, hst)
plot(bins, Fun(bins, *fitted_params), 'r')
