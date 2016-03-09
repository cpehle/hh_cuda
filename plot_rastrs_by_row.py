# -*- coding: utf-8
'''
Created on 06.11.2013

@author: Esir Pavel
'''
import csv
import numpy as np
import matplotlib
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'DejaVu Serif'
rcParams['font.size'] = 24.
rcParams['axes.labelsize'] = 26.
rcParams['lines.linewidth'] = 1.74

import matplotlib.pyplot as pl
#pl.ioff()
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
h = 0.02
varParam = np.arange(1.99, 2.021, 0.01)[:1]
varParam = np.arange(1.95, 2.041, 0.01)
#varParam = [1.97]
varParam = [1.92, 1.96, 1.97, 1.98, 2.0]

res_path = '/media/ssd/'
#res_path = './'

if len(sys.argv) > 1:
    res_path = sys.argv[1]

if len(sys.argv) > 2:
    Ie = float(sys.argv[2])

if len(sys.argv) > 3:
    rate = float(sys.argv[3])

path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}_h_{}_/seed_{}'.format(N, rate, w_n, Ie, h, seedIdx)
#path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}_h_{}_/seed_{}'.format(N, rate, w_n, Ie, h, seedIdx)
#path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}/seed_{}'.format(N, rate, w_n, Ie, seedIdx)

fig = pl.figure(figsize=(10, 10), dpi=60)
gs = GridSpec(len(varParam), 2, width_ratios = [1, 4])

num = ['I', 'II', 'III', 'IV', 'V']
for idx, param in enumerate(varParam):
    (time, awsr) = np.load(path + '/awsr_w_p_{:.3f}'.format(param) + '.npy')
    time /= 1000.
    if idx == 0:
        ax = pl.subplot(gs[idx, 1])
    else:
        ax = pl.subplot(gs[idx, 1])#, sharex=ax)

    ax2 = pl.subplot(gs[idx, 0], sharey=ax)

    ax.plot(time[::10], awsr[::10], color = 'k')
    ax.set_xticks([])
    ax.set_yticks([0., 50., 100.])
    ax.set_ylim([0., 130.])

    ax.text(0.5, 1.13, r"({}) $ w_p={} \ pA/cm^2$".format(num[idx], param),
            horizontalalignment = 'center', transform = ax.transAxes)

    ax2.hist(awsr, bins=40, orientation='horizontal',
             edgecolor='none', color='k', range=(0, 130))

    ax2.set_xticks([])
    pl.setp(ax2.get_yticklabels(), visible=False)

    if idx == 2:
        ax.set_ylabel("Firing rate")
        ax2.set_ylabel("Firing rate distributions")
        ax.add_patch(Rectangle((455, 0), 70, 130, facecolor = '#44ff44'))
        ax.add_patch(Rectangle((1205, 0), 70, 130, facecolor = '#4444ff'))

ax.set_xticks(np.linspace(0, max(time), 6))
ax.set_xlabel("$t, s$")

pl.subplots_adjust(left = 0.07, right = 0.93, top = 0.93, bottom = 0.08, wspace = 0.31, hspace = 0.52)
#pl.savefig("awsr_range.eps")
#pl.savefig("awsr_range.png", dpi=600)

#pl.show()
