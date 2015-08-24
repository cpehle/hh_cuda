# -*- coding: utf-8
'''
Created on 06.11.2013

@author: Esir Pavel
'''
import csv
import numpy as np
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
seedIdx = rank

BinSize = 20.3

Ie=5.27
N = 100
rate = 185.0
w_n = 1.3
#varParam = np.arange(2.0, 2.141, 0.01)
varParam = np.arange(2.1, 2.101, 0.01)

res_path = 'transition/'

if len(sys.argv) > 1:
    res_path = sys.argv[1]

if len(sys.argv) > 2:
    Ie = float(sys.argv[2])

if len(sys.argv) > 3:
    rate = float(sys.argv[3])

path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}/seed_{}'.format(N, rate, w_n, Ie, seedIdx)

for idx, w_p in enumerate(varParam):
    times = []
    print w_p

#    f = open('{}/w_p_{:.3f}'.format(path, w_p), "r")
#    rdr = csv.reader(f, delimiter="\t")
#    for l in rdr:
#        times.append(l[1])
#    f.close()
#    times = np.array(times, dtype="float32")

    times = np.fromfile('{}/w_p_{:.3f}_times'.format(path, w_p), dtype='float32')

    Tmax = np.max(times)
    rhist = np.histogram(times,  bins=Tmax/BinSize, range=(0, Tmax))
    time, awsr = rhist[1][:-1], rhist[0]
    np.save(path + '/awsr_w_p_{:.3f}.npy'.format(float(w_p)), (time, awsr))
