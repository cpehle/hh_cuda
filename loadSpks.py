# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:11:29 2015

@author: Pavel Esir
"""

import matplotlib.pylab as pl
import numpy as np
import csv
import os

seed = 0
Ie=6.0

N = 2
rate = 185.0
w_n = 5.4
varParam = np.arange(0.5, 2.6, 0.1)

#N = 100
#rate = 180.0
#w_n = 1.3
#varParam = np.arange(2.0, 2.15, 0.01)

#N = 1
#rate = 170.0
#w_n = 2.4
#varParam = np.arange(1.0, 2.15, 0.05)
#varParam = arange(1.0, 61.0, 5.0)

def loadIsi(w_p):
    print w_p
    isiAll = []
    for seed in range(0, 1):
        snd = []
        tm = []
        path = 'N_{N}_rate_{rate}_w_n_{w_n}_Ie_{Ie:.2f}/seed_{seed}'.format(N=N, Ie=Ie, rate=rate, seed=seed, w_n=w_n)
        fname = path + "/spkTimes_w_p_{0:.3f}.npy".format(w_p)
        if not os.path.exists(fname):
            f = open(path+"/w_p_{0:.3f}".format(w_p), "r")
            rdr = csv.reader(f,delimiter="\t")
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
            np.save(fname, times)
        else:
            isi = [[]]*N
            times = np.load(fname)
            for i in xrange(N):
                isi[i] = np.diff(times[i])

        for i in isi:
            isiAll.extend(i)
    return isiAll


isiStd = []
isiMn = []
qual = []
for w_p in varParam:
    isi = loadIsi(w_p)
#    hist(isi, histtype='step', bins=100, range=(0, 30))

#    hs = histogram(isi, bins=1000, range=(0, 200), normed=True)
    hs = histogram(isi, bins=100, range=(0, 30))
    isiHst = hs[0]
    Trange = hs[1][:-1]
#    isiHst = gs_filter(isiHst, 2)

    tmax = Trange[argmax(isiHst)]
    atmax = max(isiHst)


    df=diff(array(isiHst > atmax/2, dtype='int'))
    st = Trange[nonzero(df == 1)[0][0]]
    stp = Trange[nonzero(df == -1)[0][0]]

    figure('isi spectras')
    plot(Trange, isiHst, label=str(w_p))
    hlines(atmax/2, st, stp)
    legend()
    qual.append(atmax*tmax/(stp - st))

    isiStd.append(np.std(isi))
    isiMn.append(np.mean(isi))
isiMn = np.array(isiMn)
isiStd = np.array(isiStd)

pl.figure("qual")
#pl.semilogx(varParam, isiStd/isiMn)
pl.plot(varParam, qual)

#save('N_{}_rate_{}_w_n_{}_Ie_{:.2f}_cv.npy'.format(N, rate, w_n, Ie),
#     (varParam, isiStd, isiMn))
