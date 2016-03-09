# coding: utf-8

import nest
from numpy import *
from pylab import *
import matplotlib.gridspec as gridspec

#paremeters
h = 0.02
T = 20.28
#T = 20.272
Nspikes = 1
T_sim = T*(Nspikes + 1)
I = 5.27
w = 1.0

dts = arange(0.02, T, 0.02)
dtPost = zeros_like(dts)
for idx, dt in enumerate(dts):
    print(dt)
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': h, 'local_num_threads': 1})

    n = nest.Create('hh_psc_alpha', params = {'I_e': I, 'E_L': -54.4, 'E_Na': 55., 'C_m': 1.,
                                             'g_K': 36., 'g_L': 0.3, 'g_Na':120.})
    # limit cycle
    nest.SetStatus(n, {'Act_h': 0.22399445261588857,
                       'Act_m': 0.91317780810919347,
                       'Inact_n': 0.57467683928571844,
                       'V_m': 32.906693266145602})
    # steady state
#    nest.SetStatus(n, {'Act_h': 0.47200077442418825,
#                       'Act_m': 0.07901232952458181,
#                       'Inact_n': 0.3719705462385153,
#                       'V_m': -61.52705139129256})

#    sg = nest.Create('spike_generator',
#                     params = {'spike_times': [dt + 10, dt + 50., 125.],
#                               'spike_weights': [7., 9., 2.]})
    stimSpkTimes = [dt]# + Nspikes*T*i for i in xrange(100)]
    sg = nest.Create('spike_generator', params = {'spike_times': stimSpkTimes})
    gen_con = nest.Connect(sg, n, syn_spec={'delay': h, 'weight': w})

    sd = nest.Create('spike_detector')
    nest.Connect(n, sd)

#    mm = nest.Create('multimeter')
#    nest.SetStatus(mm, params = {'interval': h, 'record_from': ['V_m', 'Act_h', 'Act_m', 'Inact_n', 'I_ex']})
#    nest.Connect(mm, n)

    nest.Simulate(T_sim)

    spike_times = nest.GetStatus(sd, 'events')[0]['times']
    if len(spike_times) > Nspikes:
        dtPost[idx] = dt + Nspikes*T - spike_times[Nspikes]
    else:
        dtPost[idx] = nan
#%%
#figure(figsize=(8, 6))
plot(dts, dtPost, label=str(Nspikes))
#plot(arange(0, T, 0.02), arange(0, T, 0.02), '--')
#xlabel(r'$\varphi_n$', fontsize=24.0); ylabel(r'$\varphi_{n+1}$', fontsize=24.0)
#gca().tick_params(axis='both', which='major', labelsize=24.)
#xlim((0, T)); ylim((0, T))

np.save('phase_map_T_{:.3f}_{:n}per.npy'.format(T, Nspikes), (dts, dtPost))
#np.save('phase_map_T_20.272_6per.npy', (dts, dtPost))

#figure(figsize=(8, 6)); plot(dts, dtPost - dts, '-o'); plot(dts, zeros_like(dtPost))
#xlabel(r'$\varphi_n$', fontsize=24.0); ylabel(r'$PRC$', fontsize=24.0)
#gca().tick_params(axis='both', which='major', labelsize=24.)
#%%
#phases = []
#for pre, post in zip(stimSpkTimes, spike_times):
#    phases.append((pre-post) % T)
#f = figure(1)
#plot(phases[:-1], phases[1:], '-o')
#%%
#f = figure()
##f = figure(figsize=(8*1.2, 4.5*1.))
#gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
#ax1 = f.add_subplot(gs[0,0])
#ax2 = f.add_subplot(gs[1,0], sharex = ax1)
#ax3 = ax2.twinx()
#events = nest.GetStatus(mm, 'events')[0]
#ax1.plot(events['times'], events['Act_m'], c='b')
#ax1.plot(events['times'], events['Act_h'], c='g')
#ax1.plot(events['times'], events['Inact_n'], c='c')
#ax1.legend(['$m$', '$h$', '$n$'], fontsize=14, loc='upper right')
#
#ax2.set_xlabel('$t$, ms')
#ax1.set_ylabel('gating variables')
#ax2.plot(events['times'], events['I_ex'], label='External currnt', c='r')
#ax3.plot(events['times'], events['V_m'], label='Membrane potential', c='k')
#ax2.set_ylabel('$I_{syn}$, pA', rotation = 90)
#ax3.set_ylabel('$V$, mV', rotation = 90)
#ax3.set_xlim([0., T_sim])
#
#lines, labels = ax2.get_legend_handles_labels()
#lines2, labels2 = ax3.get_legend_handles_labels()
#ax2.legend(lines + lines2, labels + labels2, fontsize=12)
