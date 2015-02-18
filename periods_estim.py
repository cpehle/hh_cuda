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

maxSr = 12
srHstBins = 10
timeBin = 20.3*5/1000. # in sec

noiseStd = np.arange(0.4, 0.55, 0.01) 

path = 'res/awsr_std_{0}.npy'

MeanPeriods = np.zeros(np.shape(noiseStd))
MeanTimesUp = np.zeros(np.shape(noiseStd))
MeanTimesDown = np.zeros(np.shape(noiseStd))
StdPeriods = np.zeros(np.shape(noiseStd))
StdTimesUp = np.zeros(np.shape(noiseStd))
StdTimesDown = np.zeros(np.shape(noiseStd))
TupRatio = np.zeros(np.shape(noiseStd))
for idx, std in enumerate(noiseStd):
    Periods = []
    TimesUp = []
    TimesDown = []
    for seed in range(0, 1):
        period, time_down, time_up = analys_trans(path.format(std), maxSr=maxSr, srHstBins=srHstBins)
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
pl.plot(noiseStd, MeanPeriods*timeBin, 'b', label="Mean Periods")
pl.plot(noiseStd, StdPeriods*timeBin, 'r', label="Std Periods")
pl.xlabel("Noise Power, pA")
pl.legend()
pl.figure(2)
pl.plot(noiseStd, MeanTimesUp*timeBin, 'b', label="Mean TimeUp")
pl.plot(noiseStd, StdTimesUp*timeBin, 'r', label="Std TimeUp")
pl.xlabel("Noise Power, pA")
pl.legend()
pl.figure(3)
pl.plot(noiseStd, MeanTimesDown*timeBin, 'b', label="Mean TimeDown")
pl.plot(noiseStd, StdTimesDown*timeBin, 'r', label="Std TimeDown")
pl.xlabel("Noise Power, pA")
pl.legend()
 
pl.figure(4)
pl.plot(noiseStd, TupRatio, label="Tup/Tsum")
pl.xlabel("Noise Power, pA")
pl.legend()
 
pl.show()

