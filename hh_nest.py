import nest
import nest.voltage_trace
import numpy as np
import matplotlib.pylab as pl

I = 5.27

Tsim = 50
h = 0.05
peak_variables = {'Act_h': 0.22399445261588857, 
                  'Act_m': 0.91317780810919347,
                  'Inact_n': 0.57467683928571844,
                  'V_m': 32.906693266145602}

nest.ResetKernel()
nest.SetKernelStatus({'resolution': h, 'local_num_threads': 1})
nest.SetDefaults('hh_psc_alpha', params={'E_L': -54.4, 'E_Na': 55., 'C_m': 1., 'g_K': 36., 'g_L': 0.3, 'g_Na': 120., 'I_e': 5.27})
nest.SetDefaults('hh_psc_alpha', params=peak_variables)
neuron = nest.Create('hh_psc_alpha')
p = nest.Create('poisson_generator')
nest.SetStatus(p, params={'rate': 10.})
#nest.Connect(p, neuron, params={'weight': 1.})
mm = nest.Create('multimeter', params={'interval': h, 'record_from': ['V_m', 'I_ex', 'Inact_n', 'Act_m', 'Act_h']})
nest.ConvergentConnect(mm, neuron)
nest.Simulate(Tsim)

mm_events = nest.GetStatus(mm)[0]['events']

import csv

f = open('res.csv', "r")
rdr = csv.reader(f,delimiter=";")
t = []
V_mean_pre = []
V_mean_post = []
Inact_n = []
Act_m = []
Act_h = []

for l in rdr:
    t.append(l[0])
    V_mean_pre.append(l[1])
    V_mean_post.append(l[2])
    Inact_n.append(l[3])
    Act_m.append(l[4])
    Act_h.append(l[5])
t = np.array(t)

ax0 = pl.subplot(211)
ax1 = pl.subplot(212, sharex=ax0)
    
ax0.plot(mm_events['times'], mm_events['V_m'])
ax0.plot(t, V_mean_pre)
ax0.legend(['nest_V_m', 'V_m'])

ax1.plot(mm_events['times'], mm_events['Inact_n'])
ax1.plot(mm_events['times'], mm_events['Act_m'])
ax1.plot(mm_events['times'], mm_events['Act_h'])
ax1.plot(t, Inact_n)
ax1.plot(t, Act_m)
ax1.plot(t, Act_h)
ax1.legend(['nest_n', 'nest_m', 'nest_h', 'n', 'm', 'h'])

pl.show()