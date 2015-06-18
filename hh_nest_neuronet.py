'''
Created on 11.12.2013

@author: Pavel Esir
'''

import csv
import sys
import nest
import nest.topology as tp
import numpy as np
import time

#param = float(sys.argv[1])

Tsim = 20000.
h = 0.1

N = 30
p_con = 0.3

I = 5.27
w_n = 1.3
T = 20.3
rate = 178.0 #Hz

nest.ResetKernel()
np.random.seed(seed=1)
nest.SetKernelStatus({'resolution': h})
# nest.SetKernelStatus({'data_path': "/home/esir_p/res"})
#nest.SetKernelStatus({'data_path': "res2"})
#nest.SetKernelStatus({'data_prefix': "w_p_{0}_".format(param)})
nest.SetKernelStatus({'overwrite_files': False})

nest.SetKernelStatus({"total_num_virtual_procs": 4})
# nest.SetKernelStatus({"local_num_threads": 8})

nest.SetDefaults('hh_psc_alpha', params={'E_L': -54.4, 'E_Na': 55., 'C_m': 1., 'g_K': 36., 'g_L': 0.3, 'g_Na': 120.,
					'Act_h': 0.223994, 'Act_m': 0.913177, 'Inact_n': 0.574676,
					'I_e': I, 'V_m': 32.906693})

pos = [[np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)] for j in xrange(N)]
l = tp.CreateLayer({'positions': pos, 'elements': 'hh_psc_alpha'})
# p = nest.Create('poisson_generator', params = {'rate': rate})
# nest.Connect(p, nest.GetLeaves(l)[0], syn_spec={'weight': param, 'delay': h})
#gnoise = nest.Create('noise_generator', params = {'std': 0.5, 'mean': 0.0})
#nest.Connect(gnoise, nest.GetLeaves(l)[0], syn_spec={'delay': h})

conn_dict = {'connection_type': 'divergent', 'mask': {'box':{'lower_left': [-1.5, -1.5, -1.5], 'upper_right': [1.5, 1.5, 1.5]}}, 'kernel': p_con, 'delays': {'linear': {'c': h, 'a': T}}, 'weights': w_n, 'allow_autapses': False}
tp.ConnectLayers(l,l,conn_dict)

con_file = open("nn_params.csv", 'w')
writer = csv.writer(con_file, delimiter=' ')
conn = nest.GetConnections()
con_file.write(str(len(conn))+"\n")
statuses = nest.GetStatus(conn)
min_el = min(nest.GetLeaves(l)[0])
for i, stat in zip(conn, statuses):
    writer.writerow([int(i[0]-min_el), int(i[1]-min_el), stat['delay']])
con_file.close()

#sd = nest.Create('spike_detector', params={"to_file": True})
#nest.Connect(nest.GetLeaves(l)[0], sd)
#
#start = time.time()
#nest.Simulate(Tsim)
#print time.time() - start
