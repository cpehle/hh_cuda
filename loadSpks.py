# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:11:29 2015

@author: Pavel Esir
"""

import matplotlib.pylab as pl
import numpy as np
import csv
import os

N = 2
rate = 185.0
seed = 0
w_n = 5.4
Ie=5.0
#varParam = np.linspace(2.0, 2.13, 7, endpoint=True)
varParam = np.linspace(1.2, 4.2, 30, endpoint=False)

def loadIsi(w_p):
    print w_p
    isiAll = []
    for seed in range(0, 1):
        snd = []
        tm = []
        path = 'N_{N}_rate_{rate}_w_n_{w_n}_Ie_{Ie}/seed_{seed}'.format(N=N, Ie=Ie, rate=rate, seed=seed, w_n=w_n)
        fname = path+"/spkTimes_w_p_{0:.2f}.npy".format(w_p)
        if not os.path.exists(fname):
            f = open(path+"/w_p_{0:.2f}".format(w_p), "r")
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
for w_p in varParam:
    isi = loadIsi(w_p)
    isiStd.append(np.std(isi))
    isiMn.append(np.mean(isi))
isiMn = np.array(isiMn)
isiStd = np.array(isiStd)

pl.plot(varParam, isiStd/isiMn)
