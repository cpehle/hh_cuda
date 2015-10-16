# -*- coding: utf-8
'''
Created on 06.11.2013

@author: Esir Pavel
'''
import csv
import numpy as np
import matplotlib
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'FreeSans'
rcParams['font.size'] = 24.
rcParams['axes.labelsize'] = 26.
rcParams['lines.linewidth'] = 1.74

import matplotlib.pyplot as pl
from matplotlib.gridspec import GridSpec
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

seedIdx = 0

Ie = 5.27
N = 100
rate = 185.0
w_n = 1.3
BinSize = 20.3

#varParam = np.arange(2.0, 2.141, 0.01)[::1]
varParam = [2.03, 2.06, 2.09, 2.12, 2.14]

res_path = '/media/ssd/'

if len(sys.argv) > 1:
    res_path = sys.argv[1]

if len(sys.argv) > 2:
    Ie = float(sys.argv[2])

if len(sys.argv) > 3:
    rate = float(sys.argv[3])

path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}_/seed_{}'.format(N, rate, w_n, Ie, seedIdx)

fig = pl.figure(figsize=(10, 10), dpi=60)
gs = GridSpec(len(varParam), 2, width_ratios = [1, 4])

for idx, param in enumerate(varParam):
    (time, awsr) = np.load(path + '/awsr_w_p_{:.3f}'.format(param) + '.npy')
    time /= 1000.

    ax = pl.subplot(gs[idx, 1])
    ax2 = pl.subplot(gs[idx, 0], sharey=ax)

    ax.plot(time[::2], awsr[::2], color = 'k')
    ax.set_xticks([])
    ax.set_yticks([0., 50., 100.])
    ax.set_ylim([0., 130.])

    ax.text(0.5, 1.13, r"$w_p={0}$, pA".format(param),
            horizontalalignment = 'center', transform = ax.transAxes)

    ax2.hist(awsr, bins=40, orientation='horizontal',
             edgecolor='none', range=(0, 130))

    ax2.set_xticks([])
    pl.setp(ax2.get_yticklabels(), visible=False)

    if idx == 2:
        ax.set_ylabel("TSR")
        ax2.set_ylabel("TSR distribution")

ax.set_xticks(np.linspace(0, max(time), 6))
ax.set_xlabel("Time, s")

pl.subplots_adjust(left = 0.07, right = 0.93, top = 0.93, bottom = 0.08, wspace = 0.31, hspace = 0.52)
pl.savefig("awsr_range.png", dpi=100)
