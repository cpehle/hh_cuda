# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 11:11:29 2015

@author: Pavel Esir
"""

import matplotlib.pylab as pl
import numpy as np
import csv
import os

N = 100
rate = 180.
seed = 0
path = 'N_{0}_rate_{1}_w_n_1.3/seed_{2}'.format(N, rate, seed)

def loadIsi(path, w_p):
    print w_p
    snd = []
    tm = []
    fname = path+"/spkTimes_w_p_{0:.2f}.npy".format(w_p)
    if not os.path.exists(fname):
        f = open(path+"/w_p_{0:.2f}".format(w_p), "r")
        rdr = csv.reader(f,delimiter=";")
        for l in rdr:
            tm.append(l[0])
            snd.append(l[1])
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

    isiAll = []
    for i in isi:
        isiAll.extend(i)
    return isiAll


isiStd = []
isiMn = []
varParam = np.linspace(2.0, 2.13, 7, endpoint=True)
for w_p in varParam:
    isi = loadIsi(path, w_p)
    isiStd.append(np.std(isi))
    isiMn.append(np.mean(isi))
isiMn = np.array(isiMn)
isiStd = np.array(isiStd)

plot(varParam, isiStd/isiMn)
