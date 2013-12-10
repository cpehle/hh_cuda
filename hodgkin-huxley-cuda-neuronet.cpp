#include <stdio.h>
#include <iostream>
#include <cstring>
#include <fstream>
#include <cmath>
#include "hodgkin-huxley-cuda-neuronet.h"

float h = 0.05f;
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

int main(){
	malloc_neur_memory();
	malloc_conn_memory();
	ini_neurs_from_file();
	init_conns_from_file();
	ofstream res_file;
	res_file.open("res.csv");
	int T_sim = SimulationTime/h;
	float V_m, n_ch, m_ch, h_ch;
	float v1, v2, v3, v4;
	float n1, n2, n3, n4;
	float m1, m2, m3, m4;
	float h1, h2, h3, h4;

	for (int t = 1; t < T_sim; t++){
		res_file << t*h << "; " << V_ms[0] << "; " << V_ms[1] << "; " << n_chs[0] << "; " << m_chs[0] << "; " << h_chs[0] << "; " << endl;
		for (int n = 0; n < Nneur; n++){
			V_m = V_ms[n];
			n_ch = n_chs[n];
			m_ch = m_chs[n];
			h_ch = h_chs[n];
			v1 = hh_Vm(V_ms[n], n_chs[n], m_chs[n], h_chs[n], I_syns[n], I_es[n]);
			n1 = hh_n_ch(V_ms[n], n_chs[n]);
			m1 = hh_m_ch(V_ms[n], m_chs[n]);
			h1 = hh_h_ch(V_ms[n], h_chs[n]);
			V_ms[n] = V_m + v1;
			n_chs[n] = n_ch + n1;
			m_chs[n] = m_ch + m1;
			h_chs[n] = h_ch + h1;

			v2 = hh_Vm(V_ms[n], n_chs[n], m_chs[n], h_chs[n], I_syns[n], I_es[n]);
			n2 = hh_n_ch(V_ms[n], n_chs[n]);
			m2 = hh_m_ch(V_ms[n], m_chs[n]);
			h2 = hh_h_ch(V_ms[n], h_chs[n]);
			V_ms[n] = V_m + (v1 + v2)/2.0f;
			n_chs[n] = n_ch + (n1 + n2)/2.0f;
			m_chs[n] = m_ch + (m1 + m2)/2.0f;
			h_chs[n] = h_ch + (h1 + h2)/2.0f;

		}
	}
	res_file.close();
	cout << "Hello!" << endl;
	return 0;
}
