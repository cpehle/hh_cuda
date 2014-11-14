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

rate = 180
path = '{0}_0.1_seed_{1}_w_n_1.3'.format(rate, seed)

BinSize = 20.3
w_ps = np.arange(2.0, 2.141, 0.01)
#w_ps = np.arange(2.14, 2.98, 0.07)
#w_ps = np.arange(2.0, 2.15, 0.0075).round(4)

for widx, w_p in enumerate(w_ps):
    print w_p
    times = []
    if str(w_p) == '1.0':
        w_p = 1
    if str(w_p) == '2.0':
        w_p = 2
    f = open('{0}/0/w_p_{1}'.format(path, w_p), "r")
    rdr = csv.reader(f,delimiter=";")
    for l in rdr:
        times.append(l[0])
    f.close()
    times = np.array(times, dtype="float")
    Tmax = np.max(times)
    rhist = np.histogram(times,  bins=Tmax/BinSize, range=(0, Tmax))
    time, awsr = rhist[1][:-1], rhist[0]
    np.save(path + '/awsr_w_p_{0}.npy'.format(float(w_p)), (time, awsr))
