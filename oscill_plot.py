# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:06:52 2015

@author: Pavel Esir
"""
import matplotlib.pylab as pl
import numpy as np
import csv
from numpy.fft import fft, fftshift
from scipy.ndimage.filters import gaussian_filter as gs_filter
from oscill_load import oscill_load

Ie=4.4

N = 1
rate = 0.0
w_n = 0.0

h = 0.5
res_path = '/media/ssd/1_15/'
path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}/'.format(N, rate, w_n, Ie)
BundSz = 400
#varParam = np.arange(1.0, 201.0, 10.)
varParam = np.arange(1.0, 16.0, 1.)
#%%
qual = np.zeros_like(varParam)
fmax = np.zeros_like(varParam)
deltaf = np.zeros_like(varParam)
for idx, D in enumerate(varParam):
#    dind = (D - 1.0)*40/(201.0 - 1.0)
    dind = (D - 1.0)*15/(16.0 - 1.0)
    Nstart = int(dind*BundSz)
    print Nstart

    for i in xrange(Nstart, Nstart + BundSz):
        t, Vm = oscill_load(path+'N_{}_oscill.csv'.format(i))
        spec = abs(fft(Vm - np.mean(Vm)))
        if i == Nstart:
            specMean = spec
        else:
            specMean += spec
    specMean /= BundSz
    specMean = fftshift(specMean)
    specMean = specMean[len(specMean)/2:]

#    specMean = gs_filter(specMean, 2)

    frange = np.linspace(0, 0.5*1000/h, len(specMean))

    fmaxInd = np.argmax(specMean)
    fmax[idx] = frange[fmaxInd]
    afmax = np.max(specMean)
    df=np.diff(np.array(specMean[:fmaxInd] > afmax/np.sqrt(2), dtype='int'))
    st = frange[np.nonzero(df == 1)[0][0]]
    df=np.diff(np.array(specMean[fmaxInd:] < afmax/np.sqrt(2), dtype='int'))
    stp = frange[fmaxInd + np.nonzero(df == 1)[0][0]]

    if idx % 1 == 0:
        pl.figure('spectras')
        pl.plot(frange, specMean, label=str(D))
        pl.hlines(afmax/np.sqrt(2), st, stp)
        pl.xlim((0, 200))
        pl.legend(loc='upper right')
        pl.show()

    deltaf[idx] = stp - st
    qual[idx] = afmax*fmax[idx]/deltaf[idx]

pl.figure("q factor")
pl.semilogx(varParam, qual, label=str(Ie))

pl.figure("fmax")
pl.semilogx(varParam, fmax, label=str(Ie))

pl.figure("deltaf")
pl.semilogx(varParam, deltaf, label=str(Ie))
#pl.legend()

#%%
#Num = 10
#fig, ax = subplots(Num, 1, sharex=True)
#for i in xrange(Num):
#    t, Vm = load(10 + i)
#    ax[i].plot(t, Vm, lw=0.5)
#    setp(ax[i].get_xticklabels(), visible=False)
#    max_yticks = 2
#    yloc = pl.MaxNLocator(max_yticks)
#    ax[i].yaxis.set_major_locator(yloc)
#xlim([0, 1000])
#ylim([-80, 40])

#ylabel("Membrane potential, mV")
#xlabel("Time, ms")
#legend()
