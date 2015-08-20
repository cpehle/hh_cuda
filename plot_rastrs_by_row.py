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
from scipy.ndimage.filters import gaussian_filter
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

seedIdx = rank

Ie=5.27
N = 100
rate = 185.0
w_n = 1.3
varParam = np.arange(2.0, 2.15, 0.01)
BinSize = 20.3

res_path = 'transition/'

if len(sys.argv) > 1:
    res_path = sys.argv[1]

if len(sys.argv) > 2:
    Ie = float(sys.argv[2])

if len(sys.argv) > 3:
    rate = float(sys.argv[3])

path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}/seed_{}'.format(N, rate, w_n, Ie, seedIdx)

fig = pl.figure("rastrs", figsize=(12, 9))
matplotlib.rc('lines', linewidth=0.75)
matplotlib.rc('font', size=12.)

gs = GridSpec(len(varParam), 2, width_ratios = [1, 4])

for idx, param in enumerate(varParam):
    (time, awsr) = np.load(path + '/awsr_w_p_{:.3f}'.format(param) + '.npy')
    #awsr = gaussian_filter(awsr, 10)
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
#    ax.set_ylabel(r"$w_p={0}$".format(param))
    ax.set_ylabel(str(param))

    ax2 = pl.subplot(gs[idx, 0], sharey=ax)

    ax2.hist(awsr, bins=20, orientation='horizontal', edgecolor='none')

    pl.setp(ax2.get_xticklabels(), visible=False)
    pl.setp(ax2.get_yticklabels(), visible=False)

pl.setp(ax.get_xticklabels(), visible=True)

ax.set_xticks(np.linspace(0, max(time), 11))
ax.set_xlabel("Time, s")

# pl.tight_layout(h_pad=0.)
#pl.subplots_adjust(left = 0.05, right = 0.95, top = 0.95, bottom = 0.1)
pl.savefig(path+"/awsr_range.png", dpi=172)

#pl.show()
