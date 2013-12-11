#include <stdio.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <cmath>
#include "hodgkin-huxley-cuda-neuronet.h"

float h = 0.1f;
float SimulationTime = 50.0f;

using namespace std;

// neuron parameters
float Cm    = 1.0f ; //  pF
float g_Na  = 120.0f; // nS
float g_K   = 36.0f;
float g_L   = .3f;
float E_K   = -77.0f;
float E_Na  = 55.0f;
float E_L   = -54.4f;

float V_peak = 20.0f;

//connection parameters
float tau_psc = 0.2f;
float exp_psc = exp(-h/tau_psc);

float hh_Vm(float V, float n_ch, float m_ch, float h_ch, float I_syn, float I_e){
	return (I_e - g_K*(V - E_K)*n_ch*n_ch*n_ch*n_ch - g_Na*(V - E_Na)*m_ch*m_ch*m_ch*h_ch - g_L*(V - E_L) - I_syn)*h/Cm;
}

float hh_n_ch(float V, float n_ch){
	return (.01f*(1.0f - n_ch)*(V + 55.0f)/(1.0f - exp(-(V + 55.0f)/10.0f)) - 0.125*n_ch*exp(-(V + 65.0f)/80.0f))*h;
}

float hh_m_ch(float V, float m_ch){
	return (0.1f*(1.0f - m_ch)*(V + 40.0f)/(1.0f - exp(-(V + 40.0f)/10.0f)) - 4.0f*m_ch*exp(-(V + 65.0f)/18.0f))*h;
}

float hh_h_ch(float V, float h_ch){
	return (.07f*(1.0f - h_ch)*exp(-(V + 65.0f)/20.0f) - h_ch/(1.0f + exp(-(V + 35.0f)/10.0f)))*h;
}

void hod_hux_RK4(int n){
	V_m = V_ms[n];
	n_ch = n_chs[n];
	m_ch = m_chs[n];
	h_ch = h_chs[n];
	v1 = hh_Vm(V_ms[n], n_chs[n], m_chs[n], h_chs[n], I_syns[n], I_es[n]);
	n1 = hh_n_ch(V_ms[n], n_chs[n]);
	m1 = hh_m_ch(V_ms[n], m_chs[n]);
	h1 = hh_h_ch(V_ms[n], h_chs[n]);
	V_ms[n] = V_m + v1/2.0f;
	n_chs[n] = n_ch + n1/2.0f;
	m_chs[n] = m_ch + m1/2.0f;
	h_chs[n] = h_ch + h1/2.0f;

	v2 = hh_Vm(V_ms[n], n_chs[n], m_chs[n], h_chs[n], I_syns[n], I_es[n]);
	n2 = hh_n_ch(V_ms[n], n_chs[n]);
	m2 = hh_m_ch(V_ms[n], m_chs[n]);
	h2 = hh_h_ch(V_ms[n], h_chs[n]);
	V_ms[n] = V_m + v2/2.0f;
	n_chs[n] = n_ch + n2/2.0f;
	m_chs[n] = m_ch + m2/2.0f;
	h_chs[n] = h_ch + h2/2.0f;

	v3 = hh_Vm(V_ms[n], n_chs[n], m_chs[n], h_chs[n], I_syns[n], I_es[n]);
	n3 = hh_n_ch(V_ms[n], n_chs[n]);
	m3 = hh_m_ch(V_ms[n], m_chs[n]);
	h3 = hh_h_ch(V_ms[n], h_chs[n]);
	V_ms[n] = V_m + v3;
	n_chs[n] = n_ch + n3;
	m_chs[n] = m_ch + m3;
	h_chs[n] = h_ch + h3;

	v4 = hh_Vm(V_ms[n], n_chs[n], m_chs[n], h_chs[n], I_syns[n], I_es[n]);
	n4 = hh_n_ch(V_ms[n], n_chs[n]);
	m4 = hh_m_ch(V_ms[n], m_chs[n]);
	h4 = hh_h_ch(V_ms[n], h_chs[n]);
	V_ms[n] = V_m + (v1 + 2.0f*v2 + 2.0f*v3 + v4)/6.0f;
	n_chs[n] = n_ch + (n1 + 2.0f*n2 + 2.0f*n3 + n4)/6.0f;
	m_chs[n] = m_ch + (m1 + 2.0f*m2 + 2.0f*m3 + m4)/6.0f;
	h_chs[n] = h_ch + (h1 + 2.0f*h2 + 2.0f*h3 + h4)/6.0f;
}

int main(){
	T_sim = SimulationTime/h;
	malloc_neur_memory();
	malloc_conn_memory();
	ini_neurs_from_file();
	init_conns_from_file();
	ofstream res_file;
	res_file.open("res.csv");
	int neur;
	for (int t = 1; t < T_sim; t++){
		res_file << t*h << "; " << V_ms[0] << "; " << V_ms[1] << "; " << n_chs[0] << "; " << m_chs[0] << "; " << h_chs[0] << "; "
				<< I_psns[0] << "; " << ys[0] << "; " << endl;

		for (int n = 0; n < Nneur; n++){
			hod_hux_RK4(n);
			// checking if there's spike on neuron
			if (V_ms[n] > V_peak && V_ms_last[n] > V_ms[n] && V_ms_last_[n] <= V_ms_last[n]){
				spike_times[spike_arr_dim*num_spikes[n] + n] = t;
				num_spikes[n]++;
//				cout << t*h << endl;
//				cout << "V_m: " << V_ms[n] << " V_m_last: " << V_ms_last[n] << " V_last_: " << V_ms_last_[n] << endl;
			}
			V_ms_last_[n] = V_ms_last[n];
			V_ms_last[n] = V_ms[n];
		}

		for (int s = 0; s < Ncon; s++){
			I_psns[s]  = (ys[s]*h + I_psns[s])*exp_psc;
			ys[s] *= exp_psc;
			neur = pre_conns[s];
			// if we processed less spikes than there is in presynaptic neuron
			// we need to check are there new spikes at this moment of time
			if (num_spk_proc[s] < num_spikes[neur]){
//				cout << "Need to check for new spikes" << endl;
				if (spike_times[spike_arr_dim*num_spk_proc[s] + neur] == t - delays[s]){
					ys[s] += (exp(1.0f)/tau_psc)*weights[s];
					num_spk_proc[s]++;
//					cout << "Spike processed! Time: " << t*h << endl;
				}
			}
		}
	}
	res_file.close();
	cout << "Hello!" << endl;
	return 0;
}
