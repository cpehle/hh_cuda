# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:37:52 2015

@author: Pavel Esir
"""
from __future__ import division
from matplotlib.pylab import *
from numpy import *
import csv

rdr = csv.reader(open('nn_params_100.csv', 'r'), delimiter=' ')
i = rdr.next()
N = int(i[0])
pre = zeros(N, dtype='int')
post = zeros(N, dtype='int')
delays = zeros(N)
for idx, i in enumerate(rdr):
    pre[idx] = i[0]
    post[idx] = i[1]
    delays[idx] = i[2]

Nneur = 100
numInp = zeros(Nneur)
for i in xrange(Nneur):
    numInp[i] = sum(array(post == i, dtype='int'))
