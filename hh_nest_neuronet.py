'''
Created on 11.12.2013

@author: postdoc3
'''

import nest
import nest.voltage_trace
import nest.topology as tp
import numpy as np
import matplotlib.pylab as pl
import csv


Tsim = 1000
h = 0.1

N = 10
p_con = 0.1

I = 5.27
w_p = 2.
w_n = 1.3
T = 20.3
rate = 3620.0/T #Hz 

np.random.seed(seed=0)
nest.ResetKernel()
nest.SetKernelStatus({'resolution': h, 'local_num_threads': 1})
nest.SetDefaults('hh_psc_alpha', params={'E_L': -54.4, 'E_Na': 55., 'C_m': 1., 'g_K': 36., 'g_L': 0.3, 'g_Na': 120., 
                                         'Act_h': 0.223994, 'Act_m': 0.913177, 'Inact_n': 0.574676, 
                                         'I_e': I, 'V_m': 32.906693})

pos = [[np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)] for j in xrange(N)]
l = tp.CreateLayer({'positions': pos, 'elements': 'hh_psc_alpha'})
p = nest.Create('poisson_generator')
nest.SetStatus(p, params = {'rate': rate})
nest.CopyModel('static_synapse', 'poisson_synapse', params = {'weight': w_p, 'delay': h})
conn_dict = {'connection_type': 'divergent', 'mask': {'box':{'lower_left': [-1.5, -1.5, -1.5], 'upper_right': [1.5, 1.5, 1.5]}}, 'kernel': p_con, 'delays': {'linear': {'c': h, 'a': T}}, 'weights': w_n, 'allow_autapses': False}
tp.ConnectLayers(l,l,conn_dict)
# nest.DivergentConnect(p, nest.GetLeaves(l)[0], model = 'poisson_synapse')

con_file = open("nn_params.csv", 'w')
writer = csv.writer(con_file, delimiter=' ')
conn = nest.GetConnections()
con_file.write(str(len(conn))+"\n")
statuses = nest.GetStatus(conn)
min_el = min(nest.GetLeaves(l)[0])
for i, stat in zip(conn, statuses):
    writer.writerow([int(i[0]-min_el), int(i[1]-min_el), stat['delay']])
con_file.close()

sd = nest.Create('spike_detector')
nest.ConvergentConnect(nest.GetLeaves(l)[0], sd)

nest.Simulate(Tsim)

events = nest.GetStatus(sd)[0]["events"]
pl.plot(events['times'], events['senders']-min_el, '.')
pl.xlabel("Time, ms")
pl.ylabel("Neuron index")
pl.title("NEST")

f = open('rastr.csv')
rdr = csv.reader(f, delimiter=';')

times = []
neurons = []
for sp in rdr:
    times.append(sp[0])
    neurons.append(sp[1])
times = np.array(times, dtype='float')
neurons = np.array(neurons, dtype='int')
pl.figure()
pl.plot(times, neurons, '.')
pl.xlabel("Time, ms")
pl.ylabel("Neuron index")
pl.title("CUDA")
pl.show()
