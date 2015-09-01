# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 00:26:39 2014

@author: pavel
"""

from __future__ import print_function, division
import matplotlib
import matplotlib.pylab as pl
import numpy as np
import sys
from transition_analys import analys_trans

maxSr = 120
srHstBins = 40
timeBin = 20.3/1000.
T = 20.3

varParam = np.arange(2.0, 2.141, 0.01)

res_path = '/media/ssd/N_100_rate_182.5_w_n_1.3_Ie_5.27_lobach/'

if len(sys.argv) > 1:
    res_path = sys.argv[1]

if len(sys.argv) > 2:
    Ie = float(sys.argv[2])

if len(sys.argv) > 3:
    rate = float(sys.argv[3])

path = res_path

var = 2.1
Periods = []
TimesUp = []
TimesDown = []
srange = np.load('../corr.npy')

#for seed in xrange(0, 150):
for seed in srange:
    period, time_down, time_up = analys_trans(path + "seed_{}/awsr_w_p_{:.3f}.npy".format(seed, var),
					      maxSr=maxSr, srHstBins=srHstBins)
    Periods.extend(period)
    TimesDown.extend(time_down)
    TimesUp.extend(time_up)
Periods = np.array(Periods)
TimesDown = np.array(TimesDown)
TimesUp = np.array(TimesUp)
Tup_ = TimesUp[np.nonzero(TimesUp > 10)]
Tdown_ = TimesDown[np.nonzero(TimesDown > 10)]
#np.save("Durations.npy", (TimesUp, TimesDown, Periods))
#%%
maxHst = 100000
binSz = 400
figure()
hist(Tup_*T, bins=maxHst/binSz, range=(0, maxHst), edgecolor='none')
xlabel("Duration, ms")
title("TumesUp")

maxHst = 20000
binSz = 400
figure()
hist(Tdown_*T, bins=maxHst/binSz, range=(0, maxHst), edgecolor='none')
xlabel("Duration, ms")
title("TumesDown")
#%%
#figure("loglog")
#hst, x = histogram(TimesUp*T, bins=maxHst/binSz, range=(0, maxHst))
#loglog(x[:-1], hst)
#xlabel("Duration, ms")
#title("TumesUp")