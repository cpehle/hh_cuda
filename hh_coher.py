import nest
import numpy as np
import sys

Tsim = 100000.
h = 0.02
I = 5.
N = 1

D = 1.
path = 'std'
script = False

if len(sys.argv) > 1:
    script = True
    D = float(sys.argv[1])

if len(sys.argv) > 2:
    path = str(sys.argv[2])

nest.ResetKernel()
nest.SetKernelStatus({'resolution': h, 'local_num_threads': 1})
nest.SetKernelStatus({'data_path': "res"})

nest.SetDefaults('hh_psc_alpha', params={'E_L': -54.4, 'E_Na': 55., 'C_m': 1., 'g_K': 36., 'g_L': 0.3, 'g_Na': 120.})
nest.SetDefaults('hh_psc_alpha', params={'Act_h': 0.434479, 'Act_m': 0.144501, 'Inact_n':0.402359, 'V_m': -55.0099})

neur = nest.Create('hh_psc_alpha', n=N, params={'I_e': I})

gnoise = nest.Create('noise_generator', params = {'std': np.sqrt(h*D)/h, "mean": 0.0, "dt": h})
nest.Connect(gnoise, neur, syn_spec={'delay': h})

sd = nest.Create('spike_detector')
nest.Connect(neur, sd)

mm = nest.Create('multimeter')
nest.SetStatus(mm, params = {'interval': h*10, 'record_from': ['V_m']})
nest.Connect(mm, neur)

nest.Simulate(Tsim)

spkTimes = nest.GetStatus(sd, 'events')[0]['times']

if script:
    np.save('{path}/D_{D}_spkTimes.npy'.format(D=D, path=path), spkTimes)
else:
    import matplotlib.pylab as pl
    Vms = nest.GetStatus(mm, 'events')[0]['V_m']
    pl.plot(Vms)
