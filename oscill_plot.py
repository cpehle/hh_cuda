# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:06:52 2015

@author: Pavel Esir
"""

import csv
from numpy.fft import fft

seed = 0
Ie=5.27

N = 2
rate = 185.0
w_n = 5.4
varParam = arange(1.1, 2.6, 0.1)

#N = 100
#rate = 180.0
#w_n = 1.3
#varParam = np.arange(2.0, 2.15, 0.01)

path = 'N_{0}_rate_{1}_w_n_{2}_Ie_{3}/'.format(N, rate, w_n, Ie)

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

#def calcQuality(Nstart, Nstop):
#    for i in xrange(Nstart, Nstop):
#        t, Vm = load(i)
#        spec = abs(fft(Vm - mean(Vm)))
#        if i == Nstart:
#            specMean = spec
#        else:
#            specMean += spec
#    specMean /= (Nstop - Nstart)
#
##    specMean = gs_filter(specMean, 2)
#
#    specMean = fftshift(specMean)
#    specMean = specMean[len(specMean)/2:]
#
#    frange = linspace(0, 0.5*1000/h, len(specMean))
#
#    fmax = frange[argmax(specMean)]
#    afmax = max(specMean)
#
#    df=diff(array(specMean > afmax/2., dtype='int'))
#    egg = frange[nonzero(df)[0][1]] - frange[nonzero(df)[0][0]]
#
#    return afmax*fmax/egg
#bs = []
#for i in range(15):
#    bs.append(calcQuality(i*2, (i+1)*2))
#
#semilogx(arange(1.1, 2.6, 0.1), bs)

t, Vm = load(100)
#Nstart = 0
#Nstop = 1
#h = 0.25
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

#plot(frange, specMean)
#xlabel("freq, Hz")
#xlabel("|S|")
#
plot(t, Vm, label='1')
ylabel("Membrane potential, mV")
xlabel("Time, ms")
legend()
