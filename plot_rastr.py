# -*- coding: utf-8
'''
Created on 06.11.2013

@author: Esir Pavel
'''
import csv
import numpy as np
import matplotlib.pyplot as pl

f = open('180_0.1_seed_0_w_n_1.3/0/w_p_2.14', "r")
BinSize = 20.3

rdr = csv.reader(f,delimiter=";")
times = []
neurons = []
for l in rdr:
    times.append(l[0])
    neurons.append(l[1])
f.close()

#pl.plot(times, neurons, '.')

times = np.array(times, dtype="float")
neurons = np.array(neurons, dtype="int")
pl.figure()
#ax00 = pl.subplot(211)
#ax00.plot(times, neurons, ".k")
#ax00.set_ylim([0, max(neurons)])
#ax00.set_title(u"cuda")
#ax00.set_ylabel(u"Номер нейрона")
#ax00.grid()

Tmax = np.max(times)
ax10 = pl.gca()
#ax10 = pl.subplot(212, sharex = ax00)
hi = ax10.hist(times, bins=Tmax/BinSize, histtype='step', color = 'b')
ax10.set_ylim(0., max(hi[0]))
ax10.set_ylabel("spikes/bin")
ax10.set_xlabel("Time, ms")
ax10.grid()
#ax10.set_title(str(idx))

pl.show()
