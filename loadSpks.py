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

Ie=5.27

N = 100
rate = 185.0
w_n = 1.3
varParam = np.arange(2.0, 2.15, 0.01)

res_path = '/media/ssd/'

def loadIsi(w_p):
    print w_p
    isiAll = []
    for seedIdx in range(0, 1):
        path = res_path + 'old/N_{}_rate_{}_w_n_{}_Ie_{:.2f}_h_0.1/seed_{}/'.format(N, rate, w_n, Ie, seedIdx)

#        tm, snd = loadSpkTimes(path + 'w_p_{:.3f}'.format(w_p))
        tm = np.fromfile('{}/w_p_{:.3f}_times'.format(path, w_p), dtype='float32')
        snd = np.fromfile('{}/w_p_{:.3f}_senders'.format(path, w_p), dtype='int32')

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
    isiHst = gs_filter(isiHst, 2)

    tmaxInd = np.argmax(isiHst)
    tmax = Trange[tmaxInd]
    atmax = np.max(isiHst)

    df=np.diff(np.array(isiHst[:tmaxInd] > atmax/np.sqrt(2), dtype='int'))
    st = Trange[np.nonzero(df == 1)[0][0]]
    df=np.diff(np.array(isiHst[tmaxInd:] < atmax/np.sqrt(2), dtype='int'))
    stp = Trange[tmaxInd + np.nonzero(df == 1)[0][0] + 1]

    qual.append(atmax*tmax/(stp - st))
    if idx % 2 == 0:
        pl.figure('isi spectras')
        pl.plot(Trange, isiHst, label='D={:.2f}'.format(10**w_p))
        pl.hlines(atmax/np.sqrt(2), st, stp)
        pl.legend()
    pl.xlabel('isi, ms', fontsize=16)

#varParam = 10**varParam

isiMn = np.array(isiMn)
isiStd = np.array(isiStd)

pl.figure('qual[isi]')
pl.plot(varParam, qual, label='Ie={}'.format(Ie))
#pl.xlabel(r'$D, pA^2$', fontsize=18)
pl.xlabel(r'$w_p, pA$', fontsize=16)
pl.ylabel(r'$\beta_{isi}$', fontsize=16)
#pl.legend()
#%%
pl.figure("cv[isi]")
pl.plot(varParam, isiStd/isiMn, label='Ie={}'.format(Ie))
#pl.xlabel(r'$D, pA^2$', fontsize=16)
pl.xlabel(r'$w_p,\ pA$', fontsize=16)
pl.ylabel(r"$CV_{isi}$", fontsize=16)
legend([r"$Ie=5.27\ rate=182.5\ w_n=1.3$"])
#pl.legend()
