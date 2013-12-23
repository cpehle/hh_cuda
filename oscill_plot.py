import numpy as np
import csv
import matplotlib.pylab as pl

f = open('oscill.csv', "r")
rdr = csv.reader(f,delimiter=";")
t = []
V_mean_pre = []
V_mean_post = []
Inact_n = []
Act_m = []
Act_h = []

for l in rdr:
    t.append(l[0])
    V_mean_post.append(l[1])
    V_mean_pre.append(l[2])
    Inact_n.append(l[3])
    Act_m.append(l[4])
    Act_h.append(l[5])
t = np.array(t)
f.close()

ax0 = pl.subplot(211)
ax1 = pl.subplot(212, sharex=ax0)
    
ax0.plot(t, V_mean_pre)
ax0.plot(t, V_mean_post)
ax0.legend(['pre', 'post'])
ax0.set_title("cuda")
ax0.set_ylabel("V_m, mV")
ax0.set_xlabel("time, ms")

ax1.plot(t, Inact_n)
ax1.plot(t, Act_m)
ax1.plot(t, Act_h)
ax1.legend(['n', 'm', 'h'])
pl.show()