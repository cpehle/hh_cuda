# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:06:52 2015

@author: Pavel Esir
"""

import csv
from numpy.fft import fft

seed = 0
Ie=5.0

N = 1
rate = 170.0
w_n = 2.4
#varParam = arange(1.6, 2.15, 0.025)
varParam = arange(1.0, 10.0, 1.0)

path = 'N_{0}_rate_{1}_w_n_{2}_Ie_{3:.2f}/'.format(N, rate, w_n, Ie)

def load(seed=0):
    t = []
    Vm = []
    f = open(path+'N_{seed}_oscill.csv'.format(seed=seed), "r")
    rdr = csv.reader(f,delimiter="\t")
    for l in rdr:
        t.append(l[0])
        Vm.append(l[1])
    f.close()

    t = array(t[1:], dtype='float32')
    Vm = array(Vm[1:], dtype='float32')
    return t, Vm

def calcQuality(Nstart, Nstop):
    for i in xrange(Nstart, Nstop):
        t, Vm = load(i)
        spec = abs(fft(Vm - mean(Vm)))
        if i == Nstart:
            specMean = spec
        else:
            specMean += spec
    specMean /= (Nstop - Nstart)

#    specMean = gs_filter(specMean, 2)

    specMean = fftshift(specMean)
    specMean = specMean[len(specMean)/2:]

    frange = linspace(0, 0.5*1000/h, len(specMean))

    fmax = frange[argmax(specMean)]
    afmax = max(specMean)

    df=diff(array(specMean > afmax/2., dtype='int'))
    egg = frange[nonzero(df)[0][1]] - frange[nonzero(df)[0][0]]

    return afmax*fmax/egg

h = 0.5

#bs = []
#for i in range(10):
#    bs.append(calcQuality(i*10, (i+1)*10))
#semilogx(varParam, bs)

#Nstart = 0
#Nstop = 200
#for i in xrange(Nstart, Nstop):
#    t, Vm = load(i)
#    spec = abs(fft(Vm - mean(Vm)))
#    if i == Nstart:
#        specMean = spec
#    else:
#        specMean += spec
#specMean /= Nstop - Nstart
#
#specMean = fftshift(specMean)
#specMean = specMean[len(specMean)/2:]
#
#frange = linspace(0, 0.5*1000/h, len(specMean))
#
#figure()
#plot(frange, specMean)
#xlabel("freq, Hz")
#xlabel("|S|")

figure()
t, Vm = load(0)
plot(t, Vm, lw=0.5)
ylabel("Membrane potential, mV")
xlabel("Time, ms")
legend()
