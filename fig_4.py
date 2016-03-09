# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 00:36:19 2015

@author: Pavel Esir
"""

from __future__ import division
from matplotlib.pylab import *
from matplotlib.gridspec import GridSpec
from numpy import *
import numpy as np
import matplotlib.pylab as pl

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'FreeSerif'
rcParams['font.size'] = 18.
rcParams['axes.labelsize'] = 18.
rcParams['lines.linewidth'] = 1.74

ion()
fig = pl.figure(figsize=(8, 6))
#fig = pl.figure()
gs = GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
ax = [[]]*3
axSpks = fig.add_subplot(gs[0, 0])
axHist = axSpks.twinx()
for i in xrange(3):
    ax[i] = fig.add_subplot(gs[3 - i, 0], sharex = axSpks)

c = 'k'
v_lw = 0.2
fs2 = 18
fs1 = 18

w_p = 1.97
h = 0.5

Tmin = 465
Tmax = 515
#Tmin = 1215
#Tmax = 1265

#path = '/media/ssd/N_100_rate_185.0_w_n_1.3_Ie_5.27_h_0.02_'
path = '/media/ssd/N_100_rate_185.0_w_n_1.3_Ie_5.27_h_0.02_'
#path = '/media/ssd/N_100_rate_185.0_w_n_1.3_Ie_5.27_startFl_151'
# times in sec
times = fromfile('{}/seed_0/w_p_{:.3f}_times'.format(path, w_p), dtype='float32')/1000
senders = fromfile('{}/seed_0/w_p_{:.3f}_senders'.format(path, w_p), dtype='int32')
indLess = nonzero((times < Tmax))[0]
times = times[indLess]
senders = senders[indLess]
indMore = nonzero((times > Tmin))[0]
times = times[indMore]
senders = senders[indMore]

axSpks.scatter(times, senders, color = 'k', s = 1)
axSpks.set_xlim([Tmin, Tmax])
setp(axSpks.get_xticklabels(), visible=False)
axSpks.set_ylim([0, 100])

axHist.hist(times, histtype = 'step', linewidth=2.0, color='g', bins = int((Tmax - Tmin)/0.02))
axSpks.set_yticks([0, 50, 100])
axHist.set_yticks([0, 50, 100])

recIdx = [60, 70, 80]
#recIdx = [0, 2, 4]
colors = ['r', 'b', 'm']
Vm = [[]]*len(recIdx)
for idx, (i, c) in enumerate(zip(recIdx, colors)):
    Vm[idx] = fromfile('{}/N_{}_oscill'.format(path, i), dtype='float32')
    Vm[idx] = Vm[idx][int(Tmin*1000/h):int(Tmax*1000/h)]
    stime = linspace(Tmin, Tmax, len(Vm[0]))
    ax[idx].plot(stime, Vm[idx], linewidth=v_lw, color=c)
    setp(ax[idx].get_xticklabels(), visible=False)
    ax[idx].set_ylim([-76, 45])
    ax[idx].set_yticks([])

    tm = times[nonzero(senders == i)[0]]
    axSpks.scatter(tm, [i]*len(tm), s=6, edgecolors=c, facecolors=c)

ax[1].set_yticks([-70, 0, 40])
setp(ax[0].get_xticklabels(), visible=True)
axSpks.set_ylabel('Neuron number', fontsize = fs1)
axHist.set_ylabel('Firing rate', fontsize = fs1)
ax[1].set_ylabel('$V_m, mV$', fontsize = fs2)
ax[0].set_xlabel('$t, s$', fontsize = fs2)

pl.subplots_adjust(left = 0.14, right = 0.86, top = 0.93, bottom = 0.12, hspace = 0.08)
pl.savefig("metast_Up.png", dpi=300)
#pl.savefig("metast_Down.png", dpi=300)
