# -*- coding: utf-8

import nest
import numpy as np
import sys

Tsim = 2000.
h = 0.1
I = 5.27
N = 2
w_n=1.4
D = 80.
rate = 180.
w_p = 2.0

#Inoise = w_p*0.2*e*rate/1000

path = 'res'
script = False

if len(sys.argv) > 1:
    script = True
    D = float(sys.argv[1])

if len(sys.argv) > 2:
    path = str(sys.argv[2])

nest.ResetKernel()
nest.SetKernelStatus({'resolution': h, 'local_num_threads': 1})
nest.SetKernelStatus({'data_path': "res"})

nest.SetDefaults('hh_psc_alpha', params={'E_L': -54.4, 'E_Na': 55., 'C_m': 1.,
                                         'g_K': 36., 'g_L': 0.3, 'g_Na': 120.})

# start from the peak
nest.SetDefaults('hh_psc_alpha', params={'Act_h': 0.223, 'Act_m': 0.913,
                                         'Inact_n': 0.574, 'V_m': 32.906})

neur = nest.Create('hh_psc_alpha', n=N, params={'I_e': I})

#nest.Connect(neur, neur[::-1], conn_spec={"rule": "one_to_one"},
#                syn_spec={"weight": w_n, "delay": 6.05})

#gnoise = nest.Create('noise_generator', params = {'std': np.sqrt(h*D)/h, "dt": h})
#nest.Connect(gnoise, neur, syn_spec={'delay': h})

p = nest.Create('poisson_generator', params={'rate': rate})
nest.Connect(p, neur, syn_spec={"weight": w_p, "delay": h})

sd = nest.Create('spike_detector')
nest.Connect(neur, sd)

if script:
    nest.Simulate(Tsim)
    spkTimes = nest.GetStatus(sd, 'events')[0]
    Times = {}
    for sender in set(spkTimes['senders']):
        Times[sender] = []
    for n, t in zip(spkTimes['senders'], spkTimes['times']):
        Times[n].append(t)

    np.save('{path}/D_{D}_spkTimes.npy'.format(D=D, path=path), Times[1])
else:
    from matplotlib.pylab import figure, plot, show, xlabel, ylabel

    mm = nest.Create('multimeter', params = {'interval': h, 'record_from': ['V_m', 'I_ex']})
    nest.Connect(mm, neur)

    nest.Simulate(Tsim)

    spkTimes = nest.GetStatus(sd, 'events')[0]['times']
    mms = nest.GetStatus(mm, 'events')[0]

    mms = nest.GetStatus(mm, "events")[0]
    sndrs = set(mms['senders'])
    Times = [[]]*len(sndrs)
    V_ms = [[]]*len(sndrs)
    Iex = [[]]*len(sndrs)
    indices = [[]]*len(sndrs)

    for idx, sender in enumerate(sndrs):
        Times[idx] = mms['times'][np.nonzero(mms['senders'] == sender)]
        V_ms[idx] = mms['V_m'][np.nonzero(mms['senders'] == sender)]
        Iex[idx] = mms['I_ex'][np.nonzero(mms['senders'] == sender)]
        indices[idx] = sender

    figure()
    plot(Times[1], V_ms[1])
    show()
