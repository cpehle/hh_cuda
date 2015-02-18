import nest
import numpy as np
import matplotlib.pylab as pl


Tsim = 1000000.
h = 0.1
w_n = 5.4
I = 5.27
NumWp = 16

noise_stds = np.linspace(0.4, 0.55, NumWp)

nest.ResetKernel()
nest.SetKernelStatus({'resolution': h, 'local_num_threads': 1})
nest.SetKernelStatus({'data_path': "res"})

nest.SetDefaults('hh_psc_alpha', params={'E_L': -54.4, 'E_Na': 55., 'C_m': 1., 'g_K': 36., 'g_L': 0.3, 'g_Na': 120.})
nest.SetDefaults('hh_psc_alpha', params={'Act_h': 0.434479, 'Act_m': 0.144501, 'Inact_n':0.402359, 'V_m': -55.0099})

neur1 = nest.Create('hh_psc_alpha', n=NumWp, params={'I_e': I})
neur2 = nest.Create('hh_psc_alpha', n=NumWp, params={'I_e': I})

nest.Connect(neur1 + neur2, neur2 + neur1, 
             conn_spec={"rule": "one_to_one"}, 
             syn_spec={"weight": w_n, "delay": 6.05})

gnoise = []
sd = []
for i in range(NumWp):
    gnoise.append(nest.Create('noise_generator', params = {'std': noise_stds[i], "mean": 0.0}))
    nest.Connect(gnoise[i], [neur1[i]]+[neur2[i]], syn_spec={'delay': h})
    sd.append(nest.Create('spike_detector', params={"to_file": True}))
    nest.Connect([neur1[i]]+[neur2[i]], sd[i])

np.save("noise_stds.npy", (noise_stds, np.array(sd)))

nest.Simulate(Tsim)
