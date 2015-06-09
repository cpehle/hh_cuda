# -*- coding: utf-8
'''
Created on 06.11.2013

@author: Esir Pavel
'''
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
from scipy.ndimage.filters import gaussian_filter as gs_filter

rate = 185.0
N=2
seed=0
Ie=5.0
w_n=5.4
path = 'N_{N}_rate_{rate}_w_n_{w_n}_Ie_{Ie}/seed_{seed}'.format(N=N, rate=rate, Ie=Ie, seed=seed, w_n=w_n)

# path = 'res'

varParam = np.linspace(1.2, 4.2, 10, endpoint=False)
#varParam = np.linspace(2.0, 2.13, 7, endpoint=True)

fig = pl.figure("rastrs", figsize=(12, 9))
#fig = pl.figure(figsize=(24, 18))
matplotlib.rc('lines', linewidth=.75)
matplotlib.rc('font', size=16.)

gs = GridSpec(len(varParam), 2, width_ratios = [1, 4])

for idx, param in enumerate(varParam):
    (time, awsr) = np.load(path + '/awsr_w_p_{0:.2f}'.format(param) + '.npy')
#     (time, awsr) = np.load(path + '/awsr_std_' + str(param) + '.npy')
    awsr = gs_filter(awsr, 5)
    time /= 1000.

    ax = pl.subplot(gs[idx, 1])
    ax.plot(time, awsr, color = 'k')
    ax.set_xticks([])
    ax.set_title(r"$w_p={0}$".format(param))

    aw_hist = np.histogram(awsr, bins=10)
    ax2 = pl.subplot(gs[idx, 0])
    ax2.barh(aw_hist[1][:-1], aw_hist[0])
    ax2.set_xticks([])
    ax2.set_yticks([])
    if idx == 2:
        ax.set_ylabel("TSR", fontsize=22.)
        ax2.set_ylabel("TSR distribution", fontsize=22.)

ax.set_xticks(np.linspace(0, max(time), 11))
ax.set_xlabel("Time, s")

# pl.tight_layout(h_pad=0.)
#pl.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.1)
#pl.savefig(path+"/awsr_range.png", dpi=172)

pl.show()
