# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:57:15 2013

@author: pavel
"""

import numpy as np
import matplotlib.pyplot as pl
import time

NumBundle = 40
BundleSz = 100
BinSize = 1000.
# less whan Tsim*Nneur/T_period
NumSpikes = 100000000

f = open("rastr.csv")
idx = 0
j = 0 
times = np.zeros(NumSpikes, dtype='float32')
neurs = np.zeros(NumSpikes, dtype='int32')
for line in f:
    if (j < NumSpikes):
            idx = line.find('; ')
            times[j] = line[:idx]
            rstr = line[idx+2:]
            neurs[j] = rstr[: rstr.find('; ')]
            j += 1
f.close()
times = times[:j]
neurs = neurs[:j]
del idx, j, f, line, rstr

arr = np.zeros((NumBundle, len(times)/15), dtype='float32')
num_spikes = np.zeros(NumBundle)
for t, neur in zip(times, neurs):
    n_bund = int(neur/BundleSz)
    arr[n_bund, num_spikes[n_bund]] = t
    num_spikes[n_bund] += 1

Tmax = max(times)
spikes = {}
for i, s_times in enumerate(arr):
    spikes[i] = s_times[:num_spikes[i]]
spikes = spikes.values()
del arr, times, neurs, num_spikes, i, s_times, t, neur, n_bund

hist(spikes[5], bins=Tmax/BinSize, range=(0., Tmax), histtype='step')
