import numpy as np
import csv
import matplotlib.pylab as pl

f = open('oscill.csv', "r")
rdr = csv.reader(f,delimiter=";")
t = []
V_m_pre = []
V_m_post = []
Inact_n = []
Act_m = []
Act_h = []
I_psn = []
y_psn = []
I_syn = []

for l in rdr:
    t.append(l[0])
    V_m_post.append(l[1])
    V_m_pre.append(l[2])
    Inact_n.append(l[3])
    Act_m.append(l[4])
    Act_h.append(l[5])
    I_psn.append(l[6])
    y_psn.append(l[7])
    I_syn.append(l[8])
t = np.array(t)
f.close()

pl.figure()
ax0 = pl.subplot(211)
ax1 = pl.subplot(212, sharex=ax0)

ax0.plot(t, V_m_pre, label='2')
ax0.plot(t, V_m_post, label='1')
ax0.legend()
ax0.set_title("cuda")
ax0.set_ylabel("V_m, mV")
ax0.set_xlabel("time, ms")

#ax1.plot(t, Inact_n, label='n')
#ax1.plot(t, Act_m, label='m')
#ax1.plot(t, Act_h, label='h')
#ax1.plot(t, I_psn, label='I_psn')
#ax1.plot(t, y_psn, label='y_psn')
ax1.plot(t, I_syn, label='I_syn')
ax1.legend()
pl.show()