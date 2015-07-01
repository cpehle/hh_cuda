# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 00:26:39 2014

@author: pavel
"""

from __future__ import print_function, division
import matplotlib
matplotlib.use("Qt4Agg")
import matplotlib.pylab as pl
import numpy as np
from transition_analys import analys_trans

maxSr = 100
srHstBins = 20
timeBin = 20.3/1000. # in sec

Ie=5.20

#N = 100
#rate = 180.0
#w_n = 1.3
#varParam = np.arange(2.0, 2.15, 0.02)

N = 30
rate = 170.0
w_n = 2.4
varParam = np.arange(1.85, 2.31, 0.025)

#N=2
#rate = 185.0
#w_n=5.4
#varParam = np.arange(1.5, 3.6, 0.1)

res_path = '/media/ssd/bistability/'

path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}/'.format(N, rate, w_n, Ie)

MeanPeriods = np.zeros(np.shape(varParam))
MeanTimesUp = np.zeros(np.shape(varParam))
MeanTimesDown = np.zeros(np.shape(varParam))
StdPeriods = np.zeros(np.shape(varParam))
StdTimesUp = np.zeros(np.shape(varParam))
StdTimesDown = np.zeros(np.shape(varParam))
TupRatio = np.zeros(np.shape(varParam))
for idx, var in enumerate(varParam):
    Periods = []
    TimesUp = []
    TimesDown = []
    for seed in range(1):
        period, time_down, time_up = analys_trans(path+"seed_{}/awsr_w_p_{:.3f}.npy".format(seed, var),
                                                  maxSr=maxSr, srHstBins=srHstBins)
        Periods.extend(period)
        TimesDown.extend(time_down)
        TimesUp.extend(time_up)
    MeanPeriods[idx] = np.mean(Periods)
    MeanTimesUp[idx] = np.mean(TimesUp)
    MeanTimesDown[idx] = np.mean(TimesDown)
    StdPeriods[idx] = np.std(Periods)
    StdTimesUp[idx] = np.std(TimesUp)
    StdTimesDown[idx] = np.std(TimesDown)
    TupRatio[idx] = np.sum(TimesUp)/np.sum(Periods)

pl.figure(1)
pl.plot(varParam, MeanPeriods*timeBin, 'b', label="Mean Periods")
pl.plot(varParam, StdPeriods*timeBin, 'r', label="Std Periods")
pl.xlabel("Noise Power, pA")
pl.legend()
pl.figure(2)
pl.plot(varParam, MeanTimesUp*timeBin, 'b', label="Mean TimeUp")
pl.plot(varParam, StdTimesUp*timeBin, 'r', label="Std TimeUp")
pl.xlabel("Noise Power, pA")
pl.legend()
pl.figure(3)
pl.plot(varParam, MeanTimesDown*timeBin, 'b', label="Mean TimeDown")
pl.plot(varParam, StdTimesDown*timeBin, 'r', label="Std TimeDown")
pl.xlabel("Noise Power, pA")
pl.legend()

pl.figure(4)
pl.plot(varParam, TupRatio, label="Tup/Tsum")
pl.xlabel("Noise Power, pA")
pl.legend()

pl.show()

