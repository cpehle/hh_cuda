# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:28:59 2015

@author: pavel
"""
import numpy as np
import matplotlib.pylab as pl

Drange = np.concatenate((np.arange(10.0, 210.0, 20.0),
                        np.arange(200.0, 640.0, 40.0)))

cv = []
Dall = []
for D in Drange:
    spkTimes = np.load('{path}/D_{D}_spkTimes.npy'.format(path='res', D=D))
    isiAll = np.diff(spkTimes)
    cv.append(np.std(isiAll)/np.mean(isiAll))
    Dall.append(D)

pl.figure(1)
pl.semilogx(Dall, cv)
pl.show()

#%%
#spkTimes = np.load('{path}/D_{D}_spkTimes.npy'.format(path='res', D=50.0))
#isiAll = np.diff(spkTimes)
#cv.append(np.std(isiAll)/np.mean(isiAll))