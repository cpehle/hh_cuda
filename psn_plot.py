# -*- coding: utf-8 -*-

'''
Created on 27.09.2012

@author: pavel
'''

import numpy as np
import matplotlib.pyplot as pl
import csv

f = open('oscill.csv', "r")
rdr = csv.reader(f,delimiter=";")
t = []
V_m = []
I_psns = []
I_syn  = []
ys = []

for l in rdr:
    t.append(l[0])
    V_m.append(l[1])
    I_psns.append(l[6])
    ys.append(l[7])
    I_syn.append(l[8])
  
t = np.array(t)
ax00 = pl.subplot(211)
ax10 = pl.subplot(212, sharex=ax00)

ax00.plot(t, V_m)
ax00.set_ylabel(u'Мембранный потенциал, мВ')
ax00.legend([u"Постсинаптический", u"Пресинаптический"], prop={'size': 10})
ax00.set_title("cuda")
ax00.grid()

I_syn = np.array(I_syn, dtype='float')
I_psns = np.array(I_psns, dtype='float')

ax10.plot(t, I_psns)
ax10.legend(["I_psn", "y"], prop={'size': 10})
ax10.set_ylabel(u"Синаптическая активность")
pl.xlabel(u"Время, мс")
ax10.grid()

pl.show()
