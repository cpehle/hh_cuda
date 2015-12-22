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
w_p = 2.00
(TimesUp, TimesDown, Periods) = np.load("trans/durat_185.0_1.3_5.27_{:.3f}.npy".format(w_p))
TimesUp = np.array(TimesUp, dtype='float32')
TimesDown = np.array(TimesDown, dtype='float32')

TimesUp *= T/1000
TimesDown *= T/1000

D = 2

maxHst = 1000
minHst = 0.1
binSz = 0.8

bins = [.1]
a = 1.02
while bins[-1] < maxHst:
    bins.append(0.1 + a*bins[-1])

hst, bins = histogram(TimesUp, bins=bins, density=True)
hst = array(hst, dtype='float')
binSizes = diff(bins)

bins = bins[:-1]
#figure()
#loglog(bins, hst, '.')
#%%
def expFun(x, *args):
    A, scale = args
    return A*exp(-scale*x)

def powFun(x, *args):
    A, scale = args
    return A/x**scale

def mixedFun(x, *args):
    A1, A2, scale1, scale2, scale3, b = args
    a = 1/(1 + exp(-scale3*(x - b)))
    return (1 - a)*A1/x**scale1 + a*A2*exp(-scale2*x)
#%%
#left hand of distribution (power law)
#bins1 = bins[:int(D/binSz)]
#hst1 = hst[:int(D/binSz)]
bins1 = bins[:find(bins > D)[0]]
hst1 = hst[:find(bins > D)[0]]

params = [1, 1.5]
fitted_params,_ = scipy.optimize.curve_fit(powFun, bins1, hst1, p0=params)
A1, scale1 = fitted_params

#figure()
#plot(bins1, hst1)
#plot(bins1, powFun(bins1, *fitted_params), 'r')
#xlabel("Duration[s]")
#title("Power law fitting")
#legend(["data", "fitted"])
#%%
# right hand of distribution (exponential)
bins2 = bins[find(bins > D)[0]:]
hst2 = hst[find(bins > D)[0]:]
params = [60, 2e-03]
fitted_params,_ = scipy.optimize.curve_fit(expFun, bins2, hst2, p0=params)
A2, scale2 = fitted_params

#figure()
#plot(bins2, hst2)
#plot(bins2, expFun(bins2, *fitted_params), 'r')
#xlabel("Duration[s]")
#title("Exponential")
#legend(["data", "fitted"])
#%%
params = [A1, A2, scale1, scale2,  1.887, -0.157]
fitted_params,_ = scipy.optimize.curve_fit(mixedFun, bins, hst, p0=params, sigma=1/binSizes)
A1, A2, scale1, scale2, scale3, b = fitted_params
#%%
figure()
loglog(bins, hst, '.')
loglog(bins, mixedFun(bins, *fitted_params), '--', linewidth=1.5, label=w_p)
#loglog(bins, mixedFun(bins, *fitted_params), '--', linewidth=1.5)
xlabel(r"$Duration\ Times\ Up,\ s$")
ylabel(r"$PDF$")
ylim([10e-6, 10])
legend(["data", "fitted"], fontsize='large')
#legend(title="$w_p$")
title("$w_p$ = {}".format(w_p))
text(0.2, 4, r'$\rm A_{{pow}}={:.3f} \ \lambda_{{pow}}={:.3f}\ A_{{exp}}={:.3f}\ \lambda_{{exp}}={:.3f}$'.format(A1, scale1, A2, scale2), fontsize='large')
text(0.2, 2, r'$\rm \lambda_s={:.3f}\ a={:.3f}$'.format(scale3, b), fontsize='large')
subplots_adjust(bottom = 0.14)

#savefig("w_p_{:.3f}.png".format(w_p))

