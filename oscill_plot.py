# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:06:52 2015

@author: Pavel Esir
"""

import csv
from numpy.fft import fft
from scipy.ndimage.filters import gaussian_filter as gs_filter

Ie=6.28

N = 1
rate = 170.0
w_n = 24.
varParam = np.arange(1.0, 31.0, 10.)[:3]

h = 0.25
res_path = '/media/ssd/'
path = res_path + 'N_{}_rate_{}_w_n_{}_Ie_{:.2f}/'.format(N, rate, w_n, Ie)

def load(seed=0):
    t = []
    Vm = []
    f = open(path+'N_{}_oscill.csv'.format(seed), "r")
    rdr = csv.reader(f,delimiter="\t")
    for l in rdr:
        t.append(l[0])
        Vm.append(l[1])
    f.close()

    t = array(t[1:], dtype='float32')
    Vm = array(Vm[1:], dtype='float32')
    return t, Vm

def calcQuality(D):
    D1 = 1.
    D2 = 61.
    NumBund = 12.
    dD = (D2 - D1)/NumBund
    BundSz = 200.
    Nstart = int(BundSz*(D - D1)/dD)
    Nstop = int(BundSz*((D - D1)/dD + 1))

    for i in xrange(Nstart, Nstop):
        t, Vm = load(i)
        spec = abs(fft(Vm - mean(Vm)))
        if i == Nstart:
            specMean = spec
        else:
            specMean += spec
    specMean /= (Nstop - Nstart)
    specMean = fftshift(specMean)
    specMean = specMean[len(specMean)/2:]

    specMean = gs_filter(specMean, 4)

    frange = linspace(0, 0.5*1000/h, len(specMean))

    fmax = frange[argmax(specMean)]
    afmax = max(specMean)


    df=diff(array(specMean > afmax/2, dtype='int'))
    st = frange[nonzero(df == 1)[0][0] + 1]
    stp = frange[nonzero(df == -1)[0][0] + 1]

    figure('spectras')
    hlines(afmax/2, st, stp)
    plot(frange, specMean, label=str(D))
    xlim((0, 200))
    legend()

    qual = afmax*fmax/(stp - st)

    return qual
#%%
bs = zeros_like(varParam)

for idx, D in enumerate(varParam):
    bs[idx]  = calcQuality(D)
figure("q factor")
semilogx(varParam, bs)

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
Num = 20
fig, ax = subplots(Num, 1, sharex=True)

for i in xrange(Num):
    t, Vm = load(i)
    ax[i].plot(t, Vm, lw=0.5)
    setp(ax[i].get_xticklabels(), visible=False)
    max_yticks = 2
    yloc = pl.MaxNLocator(max_yticks)
    ax[i].yaxis.set_major_locator(yloc)
#xlim([0, 1000])
#ylim([-80, 40])

#ylabel("Membrane potential, mV")
#xlabel("Time, ms")
#legend()
