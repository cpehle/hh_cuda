# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:11:29 2015

@author: Pavel Esir
"""

import matplotlib.pylab as pl
import numpy as np
import csv
from scipy.ndimage.filters import gaussian_filter as gs_filter
from data_load import loadSpkTimes

pl.ion()
Ie=4.4

#N = 30
#rate = 170.0
#w_n = 2.4
#varParam = np.arange(1.85, 2.31, 0.025)

#N = 2
#rate = 185.0
#w_n = 5.4
#varParam = np.arange(0.5, 2.6, 0.1)

#N = 100
#rate = 180.0
#w_n = 1.3
#varParam = np.arange(2.0, 2.15, 0.01)

N = 1
rate = 0.0
w_n = 0.0
varParam = np.round(np.linspace(-0.4, 2.8, 16, endpoint=False), 1) + 0.0

res_path = '/media/ssd/noscill/'

def loadIsi(w_p):
    print w_p
    isiAll = []
    for seedIdx in range(0, 400):
        path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}/seed_{}/'.format(N, rate, w_n, Ie, seedIdx)

        tm, snd = loadSpkTimes(path + 'w_p_{:.3f}'.format(w_p))

        times = [[]]*N
        isi = [[]]*N
        for i in xrange(N):
            times[i] = tm[np.nonzero(snd == i)]
            isi[i] = np.diff(times[i])
        del snd, tm

        for i in isi:
            isiAll.extend(i)
    return isiAll

isiStd = []
isiMn = []
qual = []
for idx, w_p in enumerate(varParam):
    isi = loadIsi(w_p)
    isiStd.append(np.std(isi))
    isiMn.append(np.mean(isi))

    hs = np.histogram(isi, bins=200, range=(0, 30))
    isiHst = hs[0]
    Trange = hs[1][:-1]
    isiHst = gs_filter(isiHst, 4)

    tmaxInd = np.argmax(isiHst)
    tmax = Trange[tmaxInd]
    atmax = np.max(isiHst)

    df=np.diff(np.array(isiHst[:tmaxInd] > atmax/np.sqrt(2), dtype='int'))
    st = Trange[np.nonzero(df == 1)[0][0]]
    df=np.diff(np.array(isiHst[tmaxInd:] < atmax/np.sqrt(2), dtype='int'))
    stp = Trange[tmaxInd + np.nonzero(df == 1)[0][0] + 1]

    qual.append(atmax*tmax/(stp - st))
    if idx % 1 == 0:
        pl.figure('isi spectras')
        pl.plot(Trange, isiHst, label=str(w_p))
        pl.hlines(atmax/np.sqrt(2), st, stp)
        pl.legend()
varParam = 10**varParam

isiMn = np.array(isiMn)
isiStd = np.array(isiStd)

pl.figure("qual[isi]")
pl.semilogx(varParam, qual, label=str(Ie))
pl.legend()
#
pl.figure("cv[isi]")
pl.semilogx(varParam, isiStd/isiMn, label=str(Ie))
pl.legend()
