# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:06:52 2015

@author: Pavel Esir
"""
import matplotlib.pylab as pl
import numpy as np
from numpy.fft import fft, fftshift
from scipy.ndimage.filters import gaussian_filter as gs_filter

Ie=4.4
N = 1
rate = 170.0
w_n = 0.0

h = 0.5
res_path = '/media/ssd/'
path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}/'.format(N, rate, w_n, Ie)
BundSz = 800
#varParam = np.arange(-0.4, 2.8, 0.2)
varParam = np.arange(1, 21, 2)
#%%
#Vm = zeros(4096)
#qual2D = np.zeros((8, 16))

qual = np.zeros_like(varParam)
fmax = np.zeros_like(varParam)
deltaf = np.zeros_like(varParam)
for idx, D in enumerate(varParam):
    dind = round((D - (1))/2)

    Nstart = int(dind*BundSz)
    print Nstart
    ErNum = 0
    for i in xrange(Nstart, Nstart + BundSz):
        validDataFl = True

        Vm = np.fromfile(path+'N_{}_oscill'.format(i), dtype='float32')

        spec = abs(fft(Vm - np.mean(Vm)))
        if spec[10] != spec[10]:
            validDataFl = False
            ErNum += 1
        if i == Nstart:
            if not validDataFl:
                specMean = np.zeros_like(Vm)
            else:
                specMean = spec
        else:
            if validDataFl:
                specMean += spec
    specMean /= BundSz - ErNum
    specMean = fftshift(specMean)
    specMean = specMean[len(specMean)/2:]

    specMean = gs_filter(specMean, 2)

    frange = np.linspace(0, 0.5*1000/h, len(specMean))

    fmaxInd = np.argmax(specMean)
    fmax[idx] = frange[fmaxInd]
    afmax = np.max(specMean)
    df=np.diff(np.array(specMean[:fmaxInd] > afmax/np.sqrt(2), dtype='int'))
    st = frange[np.nonzero(df == 1)[0][0]]
    df=np.diff(np.array(specMean[fmaxInd:] < afmax/np.sqrt(2), dtype='int'))
    stp = frange[fmaxInd + np.nonzero(df == 1)[0][0] + 1]

    if idx % 2 == 0:
        pl.figure('spectras')
        pl.plot(frange, specMean, label='D={:.2f}'.format(D))
        pl.hlines(afmax/np.sqrt(2), st, stp)
        pl.xlim((0, 200))
        pl.legend(loc='upper right')
        pl.show()

    deltaf[idx] = stp - st
    qual[idx] = afmax*fmax[idx]/deltaf[idx]
#varParam = 10**varParam
#%%
pl.figure('beta')
pl.semilogx(varParam, qual, label='Ie={}'.format(Ie))
pl.xlabel(r'$D, pA^2$', fontsize=18)
pl.ylabel(r'$\beta$', fontsize=18)
pl.legend()
#%%
#pl.figure("fmax")
#pl.semilogx(varParam, fmax, label=str(Ie))
#pl.xlabel("D, pA^2", fontsize=18)
#pl.ylabel(r'$f_0$', fontsize=18)
#
#pl.figure("deltaf")
#pl.semilogx(varParam, deltaf, label=str(Ie))
#pl.xlabel("D, pA^2", fontsize=18)
#pl.ylabel(r'$\Delta f$', fontsize=18)
#
#pl.figure("h")
#pl.semilogx(varParam, qual/(fmax/deltaf), label=str(Ie))
#pl.xlabel("D, pA^2", fontsize=18)
#pl.ylabel('h', fontsize=18)
#
#pl.figure("Q")
#pl.semilogx(varParam, fmax/deltaf, label=str(Ie))
#pl.xlabel("D, pA^2", fontsize=18)
#pl.ylabel("$Q$", fontsize=18)
#%%
#Num = 20
#fig, ax = subplots(Num, 1, sharex=True, sharey=True)
#for i in xrange(Num):
#    Vm = np.fromfile(path + 'N_{}_oscill'.format(0+i), dtype='float32')
#    ax[i].plot(Vm, lw=0.5)
#    setp(ax[i].get_xticklabels(), visible=False)
#    max_yticks = 2
#    yloc = pl.MaxNLocator(max_yticks)
#    ax[i].yaxis.set_major_locator(yloc)
#setp(ax[-1].get_xticklabels(), visible=True)
#ylim([-80, 40])
#
##ylabel("Membrane potential, mV")
#xlabel("Time, ms")
#legend()
