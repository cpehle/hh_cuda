# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 00:26:39 2014

@author: pavel
"""

from __future__ import print_function, division
import matplotlib.pylab as pl
from matplotlib.pylab import *
import numpy as np
from numpy import *
import sys
from transition_analys import analys_trans

maxSr = 130
srHstBins = 40
BinSize = 20.
N = 100
Ie=5.27
rate = 185.0
w_n = 1.3

res_path = '/media/data/'
#res_path = '/media/ssd/'

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
w_p = 1.99

#path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}_h_0.02_long/'.format(N, rate, w_n, Ie)
#path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}_h_0.02_/'.format(N, rate, w_n, Ie)
path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}/'.format(N, rate, w_n, Ie)

Periods = []
TimesUp = []
TimesDown = []
ActUp = []
ActDown = []

for seed in xrange(0, 100):
    period, time_down, time_up, actdown, actup = analys_trans(path + "seed_{}/awsr_w_p_{:.3f}.npy".format(seed, w_p),
					      maxSr=maxSr, srHstBins=srHstBins)
    Periods.extend(period)
    TimesDown.extend(time_down)
    TimesUp.extend(time_up)
    ActDown.extend(actdown)
    ActUp.extend(actup)

Periods = np.array(Periods, dtype='float')*BinSize/1000
TimesDown = np.array(TimesDown, dtype='float')*BinSize/1000
TimesUp = np.array(TimesUp, dtype='float')*BinSize/1000
#%%
figure()
plot(TimesUp, ActUp, '.', label='Up')
plot(TimesDown, ActDown, '.', label='Down')
xlabel("Time Up/Down, s")
ylabel("Spikes in 20 ms")
legend()
#%%
figure()
Tmax = 2
Tmin = 0.02

hst = histogram(TimesUp, bins=int((Tmax - Tmin)/0.08), range=(Tmin, Tmax))
bins = hst[1][:-1]
hst = hst[0]
#semilogy(bins, hst, '-o')
loglog(bins, hst, '-o', label='Up')

hstDown = histogram(TimesDown, bins=int((Tmax - Tmin)/0.08), range=(Tmin, Tmax))
binsDown = hstDown[1][:-1]
hstDown = hstDown[0]
#semilogy(binsDown, hstDown, '-o')
loglog(binsDown, hstDown, '-o', label='Down')
xlabel("Time Up/Down, s")
legend()
#%%
figure()
Tmin = 2
Tmax = 800
hst = histogram(TimesUp, bins=int((Tmax - Tmin)/50), range=(Tmin, Tmax))
bins = hst[1][:-1]
hst = hst[0]
semilogy(bins, hst, '-o', label='Up')
#loglog(bins, hst, '-o', label='Up')

hstDown = histogram(TimesDown, bins=int((Tmax - Tmin)/50), range=(Tmin, Tmax))
binsDown = hstDown[1][:-1]
hstDown = hstDown[0]
semilogy(binsDown, hstDown, '-o', label='Down')
#loglog(binsDown, hstDown, '-o', label='Down')
xlabel("Time Up/Down, s")
legend()


#np.save("{}/durat_{}_{}_{}_{:.3f}.npy".format(path, rate, w_n, Ie, w_p), (TimesUp, TimesDown, Periods))
