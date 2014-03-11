import numpy as np
import csv
import matplotlib.pylab as pl

f = open('oscill.csv', "r")
rdr = csv.reader(f,delimiter=";")
t = []
V_m_1 = []
V_m_2 = []
V_m_3 = []
Inact_n = []
Act_m = []
Act_h = []
I_psn = []
y_psn = []
I_syn1 = []
I_syn2 = []
I_syn3 = []

for l in rdr:
    t.append(l[0])
    V_m_1.append(l[1])
    V_m_2.append(l[2])
    Inact_n.append(l[3])
    Act_m.append(l[4])
    Act_h.append(l[5])
    I_psn.append(l[6])
    y_psn.append(l[7])
    I_syn1.append(l[8])
    I_syn2.append(l[9])
    I_syn3.append(l[10])
    V_m_3.append(l[11])

f.close()

pl.figure()
ax0 = pl.subplot(211)
ax1 = pl.subplot(212, sharex=ax0)

ax0.plot(t, V_m_1, label='1')
ax0.plot(t, V_m_2, label='2')
ax0.plot(t, V_m_3, label='3')
ax0.legend()
ax0.set_title("cuda")
ax0.set_ylabel("Membrane potential, mV")
ax0.set_xlabel("Time, ms")
ax0.set_ylim([-80., 40.])

#ax1.plot(t, Inact_n, label='n')
#ax1.plot(t, Act_m, label='m')
#ax1.plot(t, Act_h, label='h')
#ax1.plot(t, I_psn, label='I_psn')
#ax1.plot(t, y_psn, label='y_psn')
ax1.plot(t, I_syn1, label='I_syn_1')
ax1.plot(t, I_syn2, label='I_syn_2')
ax1.plot(t, I_syn3, label='I_syn_3')
ax1.legend()
pl.show()