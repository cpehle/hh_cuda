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

seed=0
Ie=5.27

rate = 185.0
N=2
w_n=5.4
varParam = np.arange(1.5, 3.6, 0.2)

#N = 100
#rate = 180.0
#w_n = 1.3
#varParam = np.arange(2.0, 2.15, 0.02)

#N = 10
#rate = 180.0
#w_n = 5.0
#varParam = np.arange(2.0, 3.0, 0.2)

#N = 30
#rate = 170.0
#w_n = 2.4
#varParam = np.arange(1.0, 2.151, 0.1)

path = 'N_{N}_rate_{rate}_w_n_{w_n}_Ie_{Ie:.2f}/seed_{seed}'.format(N=N, rate=rate, Ie=Ie, seed=seed, w_n=w_n)

fig = pl.figure("rastrs", figsize=(12, 9))
matplotlib.rc('lines', linewidth=0.75)
matplotlib.rc('font', size=12.)

gs = GridSpec(len(varParam), 2, width_ratios = [1, 4])

for idx, param in enumerate(varParam):
    (time, awsr) = np.load(path + '/awsr_w_p_{0:.2f}'.format(param) + '.npy')
#     (time, awsr) = np.load(path + '/awsr_std_' + str(param) + '.npy')
    awsr = gs_filter(awsr, 1)
    time /= 1000.

    if idx == 0:
        ax = pl.subplot(gs[idx, 1])
        max_yticks = 2
        yloc = pl.MaxNLocator(max_yticks)
        ax.yaxis.set_major_locator(yloc)
    else:
        ax = pl.subplot(gs[idx, 1], sharey=ax, sharex=ax)
        pl.setp(ax.get_yticklabels(), visible=False)

    pl.setp(ax.get_xticklabels(), visible=False)
    ax.plot(time, awsr, color = 'k')
#    ax.set_title(r"$w_p={0}$".format(param))
    ax.set_ylabel(r"$w_p={0}$".format(param))

    ax2 = pl.subplot(gs[idx, 0], sharey=ax)

    ax2.hist(awsr, bins=20, orientation='horizontal', edgecolor='none')

    pl.setp(ax2.get_xticklabels(), visible=False)
    pl.setp(ax2.get_yticklabels(), visible=False)

pl.setp(ax.get_xticklabels(), visible=True)

#    if idx == 2:
#        ax.set_ylabel("TSR", fontsize=22.)
#        ax2.set_ylabel("TSR distribution", fontsize=22.)

ax.set_xticks(np.linspace(0, max(time), 11))
ax.set_xlabel("Time, s")

# pl.tight_layout(h_pad=0.)
#pl.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.1)
#pl.savefig(path+"/awsr_range.png", dpi=172)

pl.show()
