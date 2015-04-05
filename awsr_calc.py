# -*- coding: utf-8
'''
Created on 06.11.2013

@author: Esir Pavel
'''
import csv
import numpy as np
import sys

seed = 0
if len(sys.argv) > 1:
    seed = sys.argv[1]

rate = 180.0
N = 100
w_n = 1.3
path = 'N_{0}_rate_{1}_w_n_{2}/seed_{3}'.format(N, rate, w_n, seed)

BinSize = 20.3
w_ps = np.linspace(2.0, 2.13, 7)
w_ps = np.arange(2.0, 2.15, 0.01)

# w_ps = np.linspace(1.0, 2.0, 15, endpoint=False)

for idx, w_p in enumerate(w_ps):
    times = []
    print w_p
    f = open('{0}/w_p_{1:.2f}'.format(path, w_p), "r")
    rdr = csv.reader(f,delimiter=";")
    for l in rdr:
        times.append(l[0])
    f.close()
    times = np.array(times, dtype="float32")
    Tmax = np.max(times)
    rhist = np.histogram(times,  bins=Tmax/BinSize, range=(0, Tmax))
    time, awsr = rhist[1][:-1], rhist[0]
    np.save(path + '/awsr_w_p_{0:.2f}.npy'.format(float(w_p)), (time, awsr))
