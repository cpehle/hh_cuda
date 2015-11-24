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

T = 20.0

(TimesUp, TimesDown, Periods) = np.load("trans/durat_185.0_1.3_5.27_1.980.npy")
TimesUp = np.array(TimesUp, dtype='float32')
TimesDown = np.array(TimesDown, dtype='float32')

TimesUp *= T/1000
TimesDown *= T/1000


D = 3

maxHst = 1000
minHst = 0.2
binSz = 0.2

#maxHst2 = 4000
#minHst2 = D
#binSz2 = 10
#
#maxHst = 4000
#minHst = 2
#binSz = 1.0

#bins1 = arange(minHst1, maxHst1, binSz1)
#bins2 = arange(minHst2, maxHst2, binSz2)
#bins = concatenate((bins1, bins2))

hst, bins = histogram(TimesUp, bins=int((maxHst-minHst)/binSz), range=(minHst, maxHst))
#hst, _ = histogram(TimesUp, bins=bins)
#hst = array(hst, dtype='float')
#hst[:len(bins1)] /= binSz1
#hst[len(bins1):] /= binSz2

bins = bins[:-1]
#figure()
#plot(bins, hst)

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
#left hand of distribution (power law)
bins1 = bins[:int(D/binSz)]
hst1 = hst[:int(D/binSz)]
params = [1, 1.5]
fitted_params,_ = scipy.optimize.curve_fit(powFun, bins1, hst1, p0=params)
A1, scale1 = fitted_params

figure()
plot(bins1, hst1)
plot(bins1, powFun(bins1, *fitted_params), 'r')
xlabel("Duration[s]")
title("Power law fitting")
legend(["data", "fitted"])
#%%
# right hand of distribution (exponential)
bins2 = bins[int(D/binSz):]
hst2 = hst[int(D/binSz):]
params = [60, 2e-03]
fitted_params,_ = scipy.optimize.curve_fit(expFun, bins2, hst2, p0=params)
A2, scale2 = fitted_params

figure()
plot(bins2, hst2)
plot(bins2, expFun(bins2, *fitted_params), 'r')
xlabel("Duration[s]")
title("Exponential")
legend(["data", "fitted"])
#%%
params = [A1, A2, scale1, scale2,  1.887, -0.157]
fitted_params,_ = scipy.optimize.curve_fit(mixedFun, bins, hst, p0=params)
A1, A2, scale1, scale2, scale3, b = fitted_params
#%%
figure()
loglog(bins, hst, '.')
loglog(bins, mixedFun(bins, *fitted_params), '--r', linewidth=1.5)
#loglog(bins, mixedFun(bins, *params), '--r', linewidth=1.5)
xlabel(r"$Duration\ Times\ Up,\ s$")
ylabel(r"$Counts$")
title("Mixed fitting")
legend(["data", "fitted"])
