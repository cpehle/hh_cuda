import nest
import nest.voltage_trace
import numpy as np
import matplotlib.pylab as pl


Tsim = 500
h = 0.1
w_n = 5.4
w_p = 1.96
rate = 178.3
I = 5.27
N = 2

nest.ResetKernel()
nest.SetKernelStatus({'resolution': h, 'local_num_threads': 1})
nest.SetDefaults('hh_psc_alpha', params={'E_L': -54.4, 'E_Na': 55., 'C_m': 1., 'g_K': 36., 'g_L': 0.3, 'g_Na': 120.})
#nest.SetDefaults('hh_psc_alpha', params={'Act_h': 0.223994, 'Act_m': 0.913177, 'Inact_n': 0.574676, 'V_m': 32.906693})
nest.SetDefaults('hh_psc_alpha', params={'Act_h': 0.434479, 'Act_m': 0.144501, 'Inact_n':0.402359, 'V_m': -55.0099})

neur = nest.Create('hh_psc_alpha', n=N, params={'I_e': I})
#nest.Connect(neur, neur[::-1], params={"weight": w_n, "delay": 6.05})

#gnrtr = nest.Create('spike_generator', params={'spike_times': [200.]})
#nest.DivergentConnect(gnrtr, [neur[0]], weight=7.8, delay=h)
#p = nest.Create('poisson_generator', params={'rate': rate})
#nest.DivergentConnect(p, neur, weight=w_p, delay=h)

mm = nest.Create('multimeter', params={'interval': h, 'record_from': ['V_m', 'I_ex', 'Inact_n', 'Act_m', 'Act_h']})
nest.ConvergentConnect(mm, neur)

nest.Simulate(Tsim)

mms = nest.GetStatus(mm, 'events')[0]

V_ms = {}
I_exs = {}
Times = {}

for sender in set(mms['senders']):
    V_ms[sender] = []
    I_exs[sender] = []
    Times[sender] = []

for n, t, V_m, I_ex in zip(mms['senders'], mms['times'], mms['V_m'], mms['I_ex']):
    V_ms[n].append(V_m)
    I_exs[n].append(I_ex)
    Times[n].append(t)

Times = Times.values()
V_ms = V_ms.values()
I_exs = I_exs.values()

ax0 = pl.subplot(211)
ax1 = pl.subplot(212, sharex=ax0)

ax0.plot(Times[0], V_ms[0], label='1')
ax0.plot(Times[1], V_ms[1], label='2')
ax0.legend()
ax0.set_title("nest")
ax0.set_ylabel("Membrane potential, mV")
ax0.set_xlabel("Time, ms")
ax0.set_ylim([-80., 40.])

ax1.plot(Times[0], I_exs[0], label='I_syn1')
ax1.plot(Times[1], I_exs[1], label='I_syn2')
ax1.legend()

pl.show()