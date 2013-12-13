#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "hodgkin-huxley-cuda-neuronet.h"

float h = 0.1f;
float SimulationTime = 10000.0f; // in ms

using namespace std;

// neuron parameters
__constant__ float Cm    = 1.0f ; //  pF
__constant__ float g_Na  = 120.0f; // nS
__constant__ float g_K   = 36.0f;
__constant__ float g_L   = .3f;
__constant__ float E_K   = -77.0f;
__constant__ float E_Na  = 55.0f;
__constant__ float E_L   = -54.4f;

__constant__ float V_peak = 20.0f;

//connection parameters
float tau_psc = 0.2f;
float exp_psc = exp(-h/tau_psc);
float w_n = 1.3f;
float w_p = 2.0f;
float exp_w_p = (exp(1.0f)/tau_psc)*w_p;
float rate = 181.0f;

__device__ float hh_Vm(float V, float n_ch, float m_ch, float h_ch, float I_syn, float I_e, float h){
	return (I_e - g_K*(V - E_K)*n_ch*n_ch*n_ch*n_ch - g_Na*(V - E_Na)*m_ch*m_ch*m_ch*h_ch - g_L*(V - E_L) + I_syn)*h/Cm;
}

__device__ float hh_n_ch(float V, float n_ch, float h){
	return (.01f*(1.0f - n_ch)*(V + 55.0f)/(1.0f - exp(-(V + 55.0f)/10.0f)) - 0.125*n_ch*exp(-(V + 65.0f)/80.0f))*h;
}

__device__ float hh_m_ch(float V, float m_ch, float h){
	return (0.1f*(1.0f - m_ch)*(V + 40.0f)/(1.0f - exp(-(V + 40.0f)/10.0f)) - 4.0f*m_ch*exp(-(V + 65.0f)/18.0f))*h;
}

__device__ float hh_h_ch(float V, float h_ch, float h){
	return (.07f*(1.0f - h_ch)*exp(-(V + 65.0f)/20.0f) - h_ch/(1.0f + exp(-(V + 35.0f)/10.0f)))*h;
}

__global__ void integrate_synapses(float* y, float* I_psc, float* I_syn, float* weight, int* delay, int* pre_conn, int* post_conn,
		int* spike_time, int* num_spike_syn, int* num_spike_neur, float t, float h, float exp_psc, int Nneur){
	int s = blockDim.x*blockIdx.x + threadIdx.x;
	I_psc[s]  = (y[s]*h + I_psc[s])*exp_psc;
	y[s] *= exp_psc;
	int neur = pre_conn[s];
	// if we processed less spikes than there is in presynaptic neuron
	// we need to check are there new spikes at this moment of time
	if (num_spike_syn[s] < num_spike_neur[neur]){
		if (spike_time[Nneur*num_spike_syn[s] + neur] == t - delay[s]){
			y[s] += weight[s];
			num_spike_syn[s]++;
		}
	}
	atomicAdd(&I_syn[post_conn[s]], I_psc[s]);
}

__global__ void integrate_neurons(
		float* V_m, float* V_m_last, float* n_ch, float* m_ch, float* h_ch,
		int* spike_time, int* num_spike_neur,
		float* I_e, float* y_psn, float* I_psn, int* psn_time, int* num_psn,
		float* I_syn, float* I_syn_last, float exp_w_p, float exp_psc,
		int Nneur, int t, float h){
		int n = blockIdx.x*blockDim.x + threadIdx.x;
		I_psn[n]  = (y_psn[n]*h + I_psn[n])*exp_psc;
		y_psn[n] *= exp_psc;
		// if where is poisson impulse on neuron
		if (psn_time[Nneur*num_psn[n] + n] == t){
			y_psn[n] += exp_w_p;
			num_psn[n]++;
		}
		I_syn[n] += I_psn[n];

		float V_mem, n_channel, m_channel, h_channel;
		float v1, v2, v3, v4;
		float n1, n2, n3, n4;
		float m1, m2, m3, m4;
		float h1, h2, h3, h4;
		V_mem = V_m[n];
		n_channel = n_ch[n];
		m_channel = m_ch[n];
		h_channel = h_ch[n];
		v1 = hh_Vm(V_m[n], n_ch[n], m_ch[n], h_ch[n], I_syn_last[n], I_e[n], h);
		n1 = hh_n_ch(V_m[n], n_ch[n], h);
		m1 = hh_m_ch(V_m[n], m_ch[n], h);
		h1 = hh_h_ch(V_m[n], h_ch[n], h);
		V_m[n] = V_mem + v1/2.0f;
		n_ch[n] = n_channel + n1/2.0f;
		m_ch[n] = m_channel + m1/2.0f;
		h_ch[n] = h_channel + h1/2.0f;

		v2 = hh_Vm(V_m[n], n_ch[n], m_ch[n], h_ch[n], (I_syn[n] + I_syn_last[n])/2.0f, I_e[n], h);
		n2 = hh_n_ch(V_m[n], n_ch[n], h);
		m2 = hh_m_ch(V_m[n], m_ch[n], h);
		h2 = hh_h_ch(V_m[n], h_ch[n], h);
		V_m[n] = V_mem + v2/2.0f;
		n_ch[n] = n_channel + n2/2.0f;
		m_ch[n] = m_channel + m2/2.0f;
		h_ch[n] = h_channel + h2/2.0f;

		v3 = hh_Vm(V_m[n], n_ch[n], m_ch[n], h_ch[n], (I_syn[n] + I_syn_last[n])/2.0f, I_e[n], h);
		n3 = hh_n_ch(V_m[n], n_ch[n], h);
		m3 = hh_m_ch(V_m[n], m_ch[n], h);
		h3 = hh_h_ch(V_m[n], h_ch[n], h);
		V_m[n] = V_mem + v3;
		n_ch[n] = n_channel + n3;
		m_ch[n] = m_channel + m3;
		h_ch[n] = h_channel + h3;

		v4 = hh_Vm(V_m[n], n_ch[n], m_ch[n], h_ch[n], I_syn[n], I_e[n], h);
		n4 = hh_n_ch(V_m[n], n_ch[n], h);
		m4 = hh_m_ch(V_m[n], m_ch[n], h);
		h4 = hh_h_ch(V_m[n], h_ch[n], h);
		V_m[n] = V_mem + (v1 + 2.0f*v2 + 2.0f*v3 + v4)/6.0f;
		n_ch[n] = n_channel + (n1 + 2.0f*n2 + 2.0f*n3 + n4)/6.0f;
		m_ch[n] = m_channel + (m1 + 2.0f*m2 + 2.0f*m3 + m4)/6.0f;
		h_ch[n] = h_channel + (h1 + 2.0f*h2 + 2.0f*h3 + h4)/6.0f;

		// checking if there's spike on neuron
		if (V_m[n] > V_peak && V_mem > V_m[n] && V_m_last[n] <= V_mem){
			spike_time[Nneur*num_spike_neur[n] + n] = t;
			num_spike_neur[n]++;
		}
		V_m_last[n] = V_mem;
		I_syn_last[n] = I_syn[n];
		I_syn[n] = 0.0f;
}

int main(){
	T_sim = SimulationTime/h;
	init_neurs_from_file();
	init_conns_from_file();
	copy2device();

	clock_t start = clock();
	for (int t = 1; t < T_sim; t++){
		integrate_neurons<<<dim3(Nneur/512), dim3(512)>>>(V_ms_dev, V_ms_last_dev, n_chs_dev, m_chs_dev, h_chs_dev, spike_times_dev, num_spikes_neur_dev,
				I_es_dev, y_psns_dev, I_psns_dev, psn_times_dev, num_psns_dev, I_syns_dev, I_syns_last_dev, exp_w_p, exp_psc, Nneur, t, h);
		cudaThreadSynchronize();
		integrate_synapses<<<dim3(Nneur/512), dim3(512)>>>(ys_dev, I_pscs_dev, I_syns_dev, weights_dev, delays_dev, pre_conns_dev, post_conns_dev,
				spike_times_dev, num_spikes_syn_dev, num_spikes_neur_dev, t, h, exp_psc, Nneur);
	}
	cudaMemcpy(spike_times, spike_times_dev, Nneur*sizeof(int)*T_sim/T_sim_part, cudaMemcpyDeviceToHost);
	cudaMemcpy(num_spikes_neur, num_spikes_neur_dev, Nneur*sizeof(int), cudaMemcpyDeviceToHost);
	float time = ((float)clock() - (float)start)*1000./CLOCKS_PER_SEC;
	cout << "Elapsed time: " << time << endl;

	save2file();
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
	int* poisson_impulse_times = new int[T_sim];
	memset(num, 0, sizeof(int)*Nneur);
	memset(poisson_impulse_times, 0, T_sim*sizeof(int));

	// number of poisson impulses
	int num_inpulses = rate*SimulationTime/1000.0f;
	for (int n = 0; n < Nneur; n++){
		for (int i = 0; i < num_inpulses; i++){
			poisson_impulse_times[get_random(T_sim)] = 1;
		}

		for (int t = 0; t < T_sim; t++){
			if (poisson_impulse_times[t]){
				psn_times[Nneur*num[n] + n] = t;
				num[n]++;
			}
		}
		memset(poisson_impulse_times, 0, T_sim*sizeof(int));
	}

	free(poisson_impulse_times);
	free(num);
}

void save2file(){
	ofstream rastr_file;
	rastr_file.open("rastr.csv");
	for (int n = 0; n < Nneur; n++){
		for (int sp_num = 0; sp_num < num_spikes_neur[n]; sp_num++){
			rastr_file << spike_times[Nneur*sp_num + n]*h << "; " << n << "; "<< endl;
		}
	}
	rastr_file.close();
}
