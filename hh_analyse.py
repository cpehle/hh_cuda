# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:28:59 2015

@author: pavel
"""
import numpy as np
import matplotlib.pylab as pl

cv = []
Dall = []
for D in np.arange(20.0, 200.1, 20.0):
    spkTimes = np.load('{path}/D_{D}_spkTimes.npy'.format(path='std2', D=D))
    isiAll = np.diff(spkTimes)
    cv.append(np.std(isiAll)/np.mean(isiAll))
    Dall.append(D)

pl.semilogx(Dall, cv)
pl.show()
