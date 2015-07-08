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
pl.ioff()

Ie=4.4

N = 1
rate = 0.0
w_n = 0.0

h = 0.24
res_path = '/media/pavel/windata/'
path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}/'.format(N, rate, w_n, Ie)
BundSz = 200
varParam = np.arange(1.0, 201.0, 5.)

def load(seed=0):
    t = []
    Vm = []
    f = open(path+'N_{}_oscill.csv'.format(seed), "r")
    rdr = csv.reader(f,delimiter="\t")
    for l in rdr:
        t.append(l[0])
        Vm.append(l[1])
    f.close()

    t = np.array(t[1:], dtype='float32')
    Vm = np.array(Vm[1:], dtype='float32')
    return t, Vm

qual = np.zeros_like(varParam)
for idx, D in enumerate(varParam):
    dind = D*40/(201.0 - 1.0)
    Nstart = int(dind*BundSz)
    print D

    for i in xrange(Nstart, Nstart + 101):
        t, Vm = load(i)
        spec = abs(fft(Vm - np.mean(Vm)))
        if i == Nstart:
            specMean = spec
        else:
            specMean += spec
    specMean /= BundSz
    specMean = fftshift(specMean)
    specMean = specMean[len(specMean)/2:]

    specMean = gs_filter(specMean, 4)

    frange = np.linspace(0, 0.5*1000/h, len(specMean))
    fmax = frange[np.argmax(specMean)]
    afmax = max(specMean)


    df=np.diff(np.array(specMean > afmax/np.sqrt(2), dtype='int'))
    st = frange[np.nonzero(df == 1)[0][0] + 1]
    stp = frange[np.nonzero(df == -1)[0][0] + 1]

    if idx % 1 == 0:
        pl.figure('spectras')
        pl.plot(frange, specMean, label=str(D))
        pl.hlines(afmax/np.sqrt(2), st, stp)
        pl.xlim((0, 200))
        pl.legend(loc='upper right')
        pl.show()

    qual[idx] = afmax*fmax/(stp - st)

pl.figure("q factor")
pl.semilogx(varParam, qual, label=str(Ie))
pl.legend()
pl.show()
#%%
#D = 21.
#D1 = 1.
#D2 = 61.
#NumBund = 12.
#dD = (D2 - D1)/NumBund
#BundSz = 200.
#
#Nstart = int(BundSz*(D - D1)/dD)
#Nstop = int(BundSz*((D - D1)/dD + 1))
#
#for i in xrange(Nstart, Nstop):
#    t, Vm = load(i)
#    spec = abs(fft(Vm - mean(Vm)))
#    if i == Nstart:
#        specMean = spec
#    else:
#        specMean += spec
#specMean /= Nstop - Nstart
#specMean = gs_filter(specMean, 1)
#
#specMean = fftshift(specMean)
#specMean = specMean[len(specMean)/2:]
#
#frange = linspace(0, 0.5*1000/h, len(specMean))
#
#figure(1)
#plot(frange, specMean, label=str(D))
#xlabel("freq, Hz")
#xlabel("|S|")
#xlim([10, 200])
#legend()
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
