# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 00:36:19 2015

@author: Pavel Esir
"""

from __future__ import division
from matplotlib.pylab import *
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
axSpks = fig.gca()
axHist = axSpks.twinx()

c = 'k'
v_lw = 0.2
fs2 = 18
fs1 = 18

w_p = 1.99
h = 0.5
sClr = ['b', 'k']
aClr = ['g', 'r']
sizes = [3, 1]
full = 0; path = '/media/ssd/N_100_rate_185.0_w_n_1.3_Ie_5.27_startFl_139'
#full = 1; path = '/media/ssd/N_100_rate_185.0_w_n_1.3_Ie_5.27_h_0.02_'

times = fromfile('{}/seed_0/w_p_{:.3f}_times'.format(path, w_p), dtype='float32')/1000
senders = fromfile('{}/seed_0/w_p_{:.3f}_senders'.format(path, w_p), dtype='int32')
Tmin = 0.
Tmax = 320.
indLess = nonzero((times < Tmax))[0]
times = times[indLess]
senders = senders[indLess]
indMore = nonzero((times > Tmin))[0]
times = times[indMore]
senders = senders[indMore]

axSpks.scatter(times, senders, color=sClr[full], s = sizes[full])
axSpks.set_ylim([0, 100])

axHist.hist(times, histtype = 'step', linewidth=2.0, color=aClr[full], bins = int((max(times) - min(times))/0.02))
axSpks.set_yticks([0, 50, 100])
axHist.set_yticks([0, 50, 100])

axSpks.set_ylabel('Neuron number', fontsize = fs1)
axHist.set_ylabel('Firing rate', fontsize = fs1)
axSpks.set_xlabel('$t, s$', fontsize = fs2)

