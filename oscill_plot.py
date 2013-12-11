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
V_mean_pre = []
V_mean_post = []
ys = []
I_psns = []

for l in rdr:
    t.append(l[0])
    V_mean_pre.append(l[1])
    V_mean_post.append(l[2])
    I_psns.append(l[6])
    ys.append(l[7])
  
t = np.array(t)
pl.figure()
ax00 = pl.subplot(211)
ax10 = pl.subplot(212, sharex=ax00)
ax00.plot(t, V_mean_pre, 'r')
ax00.plot(t, V_mean_post, 'b')
ax00.set_ylabel(u'Мембранный потенциал, мВ')
ax00.legend([u"Пресинаптический", u"Постсинаптический"], prop={'size': 10})
ax00.grid()
ax10.plot(t, I_psns)
# ax10.plot(t, ys)
ax10.legend(["I_psn", "y"], prop={'size': 10})
ax10.set_ylabel(u"Синаптическая активность")
pl.xlabel(u"Время, мс")
ax10.grid()

pl.show()
