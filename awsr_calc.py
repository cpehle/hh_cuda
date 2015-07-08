# -*- coding: utf-8
'''
Created on 06.11.2013

@author: Esir Pavel
'''
import csv
import numpy as np
import sys

seedIdx = 0
if len(sys.argv) > 1:
    seedIdx = sys.argv[1]

BinSize = 20.3
Ie=5.27

#N = 2
#rate = 185.0
#w_n = 5.4
#varParam = np.arange(1.5, 3.6, 0.1)

#N = 10
#rate = 180.0
#w_n = 5.0
#varParam = np.arange(1.0, 5.0, 0.1)

N = 30
rate = 170.0
w_n = 2.4
varParam = np.arange(1.5, 2.5, 0.05)
#varParam = np.arange(1.6, 2.1251, 0.025)

#N = 100
#rate = 185.0
#w_n = 1.3
#varParam = np.arange(2.0, 2.15, 0.01)

res_path = ''

path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}/seed_{}'.format(N, rate, w_n, Ie, seedIdx)

for idx, w_p in enumerate(varParam):
    times = []
    print w_p
    f = open('{}/w_p_{:.3f}'.format(path, w_p), "r")
    rdr = csv.reader(f, delimiter="\t")
    for l in rdr:
        times.append(l[1])
    f.close()
    times = np.array(times, dtype="float32")
    Tmax = np.max(times)
    rhist = np.histogram(times,  bins=Tmax/BinSize, range=(0, Tmax))
    time, awsr = rhist[1][:-1], rhist[0]
    np.save(path + '/awsr_w_p_{:.3f}.npy'.format(float(w_p)), (time, awsr))
