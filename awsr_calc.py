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

rate = 185.0
N = 2
w_n = 5.4
Ie=5.4
path = 'N_{0}_rate_{1}_w_n_{2}_Ie_{3}/seed_{4}'.format(N, rate, w_n, Ie, seed)

BinSize = 20.3
w_ps = np.arange(1.4, 4.2, 0.1)
#w_ps = np.arange(2.0, 2.15, 0.01)

# w_ps = np.linspace(1.0, 2.0, 15, endpoint=False)

for idx, w_p in enumerate(w_ps):
    times = []
    print w_p
    f = open('{0}/w_p_{1:.2f}'.format(path, w_p), "r")
    rdr = csv.reader(f,delimiter="\t")
    for l in rdr:
        times.append(l[1])
    f.close()
    times = np.array(times, dtype="float32")
    Tmax = np.max(times)
    rhist = np.histogram(times,  bins=Tmax/BinSize, range=(0, Tmax))
    time, awsr = rhist[1][:-1], rhist[0]
    np.save(path + '/awsr_w_p_{0:.2f}.npy'.format(float(w_p)), (time, awsr))
