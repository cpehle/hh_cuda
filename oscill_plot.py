import numpy as np
import csv
import matplotlib.pylab as pl

f = open('oscill.csv', "r")
rdr = csv.reader(f,delimiter=";")
t = []
V_m_1 = []
V_m_2 = []
I_psn_1 = []
I_psn_2 = []
I_syn_1 = []
I_syn_2 = []


for l in rdr:
    t.append(l[0])
    V_m_1.append(l[1])
    V_m_2.append(l[2])
    I_psn_1.append(l[3])
    I_psn_2.append(l[4])
    I_syn_1.append(l[5])
    I_syn_2.append(l[6])

f.close()

pl.figure()
ax0 = pl.subplot(211)
ax1 = pl.subplot(212, sharex=ax0)

ax0.plot(t, V_m_1, label='1')
ax0.plot(t, V_m_2, label='2')
ax0.legend()
ax0.set_title("cuda")
ax0.set_ylabel("Membrane potential, mV")
ax0.set_xlabel("Time, ms")

ax1.plot(t, I_psn_1, label='I_psn_1')
ax1.plot(t, I_psn_2, label='I_psn_2')
ax1.plot(t, I_syn_1, '--',label='I_syn_1')
ax1.plot(t, I_syn_2, '--',label='I_syn_2')
ax1.legend()
pl.show()