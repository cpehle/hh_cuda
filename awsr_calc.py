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
Ie=5.27

#N = 2
#rate = 185.0
#w_n = 5.4
#varParam = np.arange(1.1, 2.6, 0.1)

N = 10
rate = 185.0
w_n = 5.0
varParam = np.arange(1.0, 5.0, 0.1)

#N = 100
#rate = 180.0
#w_n = 1.3
#varParam = np.arange(2.0, 2.15, 0.01)

path = 'N_{0}_rate_{1}_w_n_{2}_Ie_{3}/seed_{4}'.format(N, rate, w_n, Ie, seed)

BinSize = 20.3

for idx, w_p in enumerate(varParam):
    times = []
    print w_p
    f = open('{0}/w_p_{1:.2f}'.format(path, w_p), "r")
    rdr = csv.reader(f,delimiter="\t")
    for l in rdr:
        times.append(l[1])
    f.close()
    times = np.array(times, dtype="float32")
#    Tmax = np.max(times)
    Tmax = 100000.
    rhist = np.histogram(times,  bins=Tmax/BinSize, range=(0, Tmax))
    time, awsr = rhist[1][:-1], rhist[0]
    np.save(path + '/awsr_w_p_{0:.2f}.npy'.format(float(w_p)), (time, awsr))
