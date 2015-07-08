# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:11:29 2015

@author: Pavel Esir
"""

import matplotlib.pylab as pl
import numpy as np
import csv
import os

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
varParam = np.arange(1.0, 201.0, 5.)

res_path = '/media/ssd/no_oscill/'

def loadIsi(w_p):
    print w_p
    isiAll = []
    for seedIdx in range(0, 100):
        path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}/seed_{}/'.format(N, rate, w_n, Ie, seedIdx)
        f = open(path + 'w_p_{:.3f}'.format(w_p), "r")

        rdr = csv.reader(f,delimiter="\t")

        snd = []
        tm = []
        for l in rdr:
            snd.append(l[0])
            tm.append(l[1])
        f.close()
        tm = np.array(tm, dtype="float32")
        snd = np.array(snd, dtype="float32")

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

    hs = histogram(isi, bins=100, range=(0, 30))
    isiHst = hs[0]
    Trange = hs[1][:-1]
    isiHst = gs_filter(isiHst, 4)

    tmax = Trange[argmax(isiHst)]
    atmax = max(isiHst)

    df=diff(array(isiHst > atmax/2, dtype='int'))
    st = Trange[nonzero(df == 1)[0][0]]
    stp = Trange[nonzero(df == -1)[0][0]]
    qual.append(atmax*tmax/(stp - st))
#    if idx % 10 == 0:
#        figure('isi spectras')
#        plot(Trange, isiHst, label=str(w_p))
#        hlines(atmax/2, st, stp)
#        legend()

isiMn = np.array(isiMn)
isiStd = np.array(isiStd)

pl.figure("qual[isi]")
pl.plot(varParam, qual, label=str(Ie))
pl.legend()

pl.figure("cv[isi]")
pl.plot(varParam, isiStd/isiMn, label=str(Ie))
pl.legend()
