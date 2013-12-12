#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "hodgkin-huxley-cuda-neuronet.h"

float h = 0.1f;
float SimulationTime = 500.0f;

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
float w_n = 1.0f;
float w_p = 2.0f;
float exp_w_p = (exp(1.0f)/tau_psc)*w_p;
float rate = 200.0f;

float hh_Vm(float V, float n_ch, float m_ch, float h_ch, float I_syn, float I_e){
	return (I_e - g_K*(V - E_K)*n_ch*n_ch*n_ch*n_ch - g_Na*(V - E_Na)*m_ch*m_ch*m_ch*h_ch - g_L*(V - E_L) + I_syn)*h/Cm;
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
	v1 = hh_Vm(V_ms[n], n_chs[n], m_chs[n], h_chs[n], I_syns_[n], I_es[n]);
	n1 = hh_n_ch(V_ms[n], n_chs[n]);
	m1 = hh_m_ch(V_ms[n], m_chs[n]);
	h1 = hh_h_ch(V_ms[n], h_chs[n]);
	V_ms[n] = V_m + v1/2.0f;
	n_chs[n] = n_ch + n1/2.0f;
	m_chs[n] = m_ch + m1/2.0f;
	h_chs[n] = h_ch + h1/2.0f;

	v2 = hh_Vm(V_ms[n], n_chs[n], m_chs[n], h_chs[n], (I_syns[n] + I_syns_[n])/2.0f, I_es[n]);
	n2 = hh_n_ch(V_ms[n], n_chs[n]);
	m2 = hh_m_ch(V_ms[n], m_chs[n]);
	h2 = hh_h_ch(V_ms[n], h_chs[n]);
	V_ms[n] = V_m + v2/2.0f;
	n_chs[n] = n_ch + n2/2.0f;
	m_chs[n] = m_ch + m2/2.0f;
	h_chs[n] = h_ch + h2/2.0f;

	v3 = hh_Vm(V_ms[n], n_chs[n], m_chs[n], h_chs[n], (I_syns[n] + I_syns_[n])/2.0f, I_es[n]);
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
	init_neurs_from_file();
	init_conns_from_file();
	ofstream oscill_file, rastr_file;
	oscill_file.open("oscill.csv");
	rastr_file.open("rastr.csv");
	clock_t start = clock();
	for (int t = 1; t < T_sim; t++){
		oscill_file << t*h << "; " << V_ms[2] << "; " << V_ms[6] << "; " << n_chs[6] << "; " << m_chs[6] << "; " << h_chs[6] << "; "
				<< I_poisson[2] << "; " << y_poisson[2] << "; " << endl;

		for (int n = 0; n < Nneur; n++){
			I_poisson[n]  = (y_poisson[n]*h + I_poisson[n])*exp_psc;
			y_poisson[n] *= exp_psc;
			// if where is poisson impulse on neuron
			if (poisson_times[Nneur*poisson_processed[n] + n] == t){
				y_poisson[n] += exp_w_p;
				poisson_processed[n]++;
			}
//			I_syns[n] += I_poisson[n];

			hod_hux_RK4(n);
			// checking if there's spike on neuron
			if (V_ms[n] > V_peak && V_m > V_ms[n] && V_ms_last[n] <= V_m){
				spike_times[Nneur*num_spikes[n] + n] = t;
				num_spikes[n]++;
				rastr_file << t*h << "; " << n << "; " << endl;
			}
			V_ms_last[n] = V_m;
			I_syns_[n] = I_syns[n];
			I_syns[n] = 0.0f;
		}

		for (int s = 0; s < Ncon; s++){
			I_psns[s]  = (ys[s]*h + I_psns[s])*exp_psc;
			ys[s] *= exp_psc;
			int neur = pre_conns[s];
			// if we processed less spikes than there is in presynaptic neuron
			// we need to check are there new spikes at this moment of time
			if (num_spk_proc[s] < num_spikes[neur]){
				if (spike_times[Nneur*num_spk_proc[s] + neur] == t - delays[s]){
					ys[s] += weights[s];
					num_spk_proc[s]++;
				}
			}
			I_syns[post_conns[s]] += I_psns[s];
		}
	}
	float time = ((float)clock() - (float)start)*1000./CLOCKS_PER_SEC;
	cout << "Elapsed time: " << time << endl;
	oscill_file.close();
	rastr_file.close();
	cout << "Finished!" << endl;
	return 0;
}

void init_conns_from_file(){
	ifstream con_file;
	con_file.open("nn_params.csv");
	con_file >> Ncon;
	cout << "Number of connections: " << Ncon << endl;
	malloc_conn_memory();
	float delay;
	for (int s = 0; s < Ncon; s++){
		con_file >> pre_conns[s] >> post_conns[s] >> delay;
		delays[s] = delay/h;
		weights[s] = (exp(1.0f)/tau_psc)*w_n;
	}
}

void init_neurs_from_file(){
	malloc_neur_memory();
	for (int n = 0; n < Nneur; n++){
		V_ms[n] = 32.9066f;
		V_ms_last[n] = 32.9065f;
		m_chs[n] = 0.913177f;
		n_chs[n] = 0.574678f;
		h_chs[n] = 0.223994f;
		I_es[n] = 5.27f;
	}
	int* num = new int[Nneur];
	int* poisson_impulse_times = new int[T_sim*Nneur];
	memset(num, 0, sizeof(int)*Nneur);
	memset(poisson_impulse_times, 0, T_sim*Nneur*sizeof(int));

	// number of poisson impulses
	int num_inpulses = rate*SimulationTime/1000.0f;
	for (int i = 0; i < num_inpulses; i++){
		for (int n = 0; n < Nneur; n++){
			poisson_impulse_times[Nneur*(get_random(T_sim)) + n] = 1;
		}
	}

	for (int t = 0; t < T_sim; t++){
		for (int n = 0; n < Nneur; n++){
			if (poisson_impulse_times[Nneur*t + n]){
				poisson_times[Nneur*num[n] + n] = t;
				num[n]++;
			}
		}
	}

	free(poisson_impulse_times);
	free(num);
}
