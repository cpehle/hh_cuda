# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 00:26:39 2014

@author: pavel
"""

from __future__ import print_function, division
#import matplotlib
#import matplotlib.pylab as pl
import numpy as np
import sys
from transition_analys import analys_trans

maxSr = 100
srHstBins = 40

N = 100
Ie=5.27
rate = 185.0
w_n = 1.3
w_p = 2.08

res_path = '/home/esir_p/hh_cuda/long_new/'

if len(sys.argv) > 1:
    res_path = sys.argv[1]

if len(sys.argv) > 2:
    Ie = float(sys.argv[2])

if len(sys.argv) > 3:
    rate = float(sys.argv[3])

#for w_p in np.arange(1.91, 2.02, 0.01):
var_range = np.arange(1.90, 2.041, 0.01)

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
w_p = var_range[rank]

path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}_h_0.02/'.format(N, rate, w_n, Ie)
#path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}/'.format(N, rate, w_n, Ie)

Periods = []
TimesUp = []
TimesDown = []

for seed in xrange(0, 600):
    period, time_down, time_up = analys_trans(path + "seed_{}/awsr_w_p_{:.3f}.npy".format(seed, w_p),
					      maxSr=maxSr, srHstBins=srHstBins)
    Periods.extend(period)
    TimesDown.extend(time_down)
    TimesUp.extend(time_up)
Periods = np.array(Periods)
TimesDown = np.array(TimesDown)
TimesUp = np.array(TimesUp)

np.save("{}/durat_{}_{}_{}_{:.3f}.npy".format(path, rate, w_n, Ie, w_p), (TimesUp, TimesDown, Periods))
