# -*- coding: utf-8
'''
Created on 06.11.2013

@author: Esir Pavel
'''
import csv
import numpy as np
# matplotlib.use('Agg')

BinSize = 20.3*5

noise_stds, sd = np.load("noise_stds.npy")
path = "res"

for param, sd_idx in zip(noise_stds, sd):
    times = []
    neurons = []
    for seed in range(4):
        f = open(path+'/spike_detector-{0}-{1}.gdf'.format(int(sd_idx), seed), "r")
        rdr = csv.reader(f,delimiter='\t')
        for l in rdr:
            neurons.append(l[0])
            times.append(l[1])
        f.close()

    times = np.array(times, dtype="float32")
    neurons = np.array(neurons, dtype="int")
    Tmax = np.max(times)
    rhist = np.histogram(times,  bins=Tmax/BinSize, range=(0, Tmax))
    time, awsr = rhist[1][:-1], rhist[0]
    np.save(path + '/awsr_std_{0}.npy'.format(float(param)), (time, awsr))
