#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "hodgkin-huxley-cuda-neuronet.h"

float h = 0.1f;
float SimulationTime = 1000.0f; // in ms
int T_sim = SimulationTime/h;
int T_sim_part = 100.0f/h;
int Nneur = 4096;

// maximum part of simulating time for which is allocated memory
// so the maximum number of spikes per neuron which can be processed is
// T_sim/T_sim_part
int T_sim_ratio = 100;
// similarly to poisson spikes for each neuron
// defined by poisson frequence
// T_sim_part_poisson = 1000/(h*max_rate)
int T_sim_part_psn = 40;

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
	ofstream rastr_file;
	rastr_file.open("rastr.csv");
	rastr_file.close();
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
		if ((t % T_sim_part) == 0){
			cout << t*h << endl;
			cudaMemcpy(spike_times, spike_times_dev, Nneur*sizeof(int)*T_sim/T_sim_ratio, cudaMemcpyDeviceToHost);
			cudaMemcpy(num_spikes_neur, num_spikes_neur_dev, Nneur*sizeof(int), cudaMemcpyDeviceToHost);
			swap_spikes();
			cudaMemcpy(spike_times_dev, spike_times, Nneur*sizeof(int)*T_sim/T_sim_ratio, cudaMemcpyHostToDevice);
			cudaMemcpy(num_spikes_neur_dev, num_spikes_neur, Nneur*sizeof(int), cudaMemcpyHostToDevice);
		}
	}
	cudaMemcpy(spike_times, spike_times_dev, Nneur*sizeof(int)*T_sim/T_sim_ratio, cudaMemcpyDeviceToHost);
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
	rastr_file.open("rastr.csv", ios_base::app);
	for (int n = 0; n < Nneur; n++){
		for (int sp_num = 0; sp_num < num_spikes_neur[n]; sp_num++){
			rastr_file << spike_times[Nneur*sp_num + n]*h << "; " << n << "; "<< endl;
		}
	}
	rastr_file.close();
}

void swap_spikes(){
	ofstream rastr_file;
	rastr_file.open("rastr.csv", ios_base::app);

	int* spike_times_temp = new int[Nneur*T_sim/T_sim_ratio];
	int* min_spike_nums_syn = new int[Nneur];
	for (int n = 0; n < Nneur; n++){
		min_spike_nums_syn[n] = INT_MAX;
	}
	for (int s = 0; s < Ncon; s++){
		if (num_spikes_syn[s] < min_spike_nums_syn[pre_conns[s]]){
			min_spike_nums_syn[pre_conns[s]] = num_spikes_syn[s];
		}
	}

	for (int n = 0; n < Nneur; n++){
		cout << "Nneur: " << n << " num_spikes: " << num_spikes_neur[n] <<
				" min_spike_num: " << min_spike_nums_syn[n] << endl;
	}

	for (int n = 0; n < Nneur; n++){
		for (int sp_n = 0; sp_n < min_spike_nums_syn[n]; sp_n++){
			rastr_file << spike_times[Nneur*sp_n + n]*h << "; " << n << "; "<< endl;
		}

		for (int sp_n = min_spike_nums_syn[n]; sp_n < num_spikes_neur[n]; sp_n++){
			spike_times_temp[Nneur*(sp_n - min_spike_nums_syn[n]) + n] = spike_times[Nneur*sp_n + n];
		}
		num_spikes_neur[n] = num_spikes_neur[n] - min_spike_nums_syn[n];
	}
	rastr_file.close();

	for (int s = 0; s < Ncon; s++){
		num_spikes_syn[s] = num_spikes_syn[s] - min_spike_nums_syn[pre_conns[s]];
	}
	free(spike_times);
	free(min_spike_nums_syn);
	spike_times = spike_times_temp;
}

//void init_poisson(int current_time){
//	int* poisson_impulse_times = new int[T_sim_cycle];
//	memset(poisson_impulse_times, 0, T_sim_cycle*sizeof(int));
//
//	// number of poisson impulses, rate in Hz
//	int num_inpulses = rate*T_sim_cycle*h/1000.0f;
//	cout << "Num impulses: " << num_inpulses << endl;
//	for (int n = 0; n < Nneur; n++){
//		for (int i = 0; i < num_inpulses; i++){
//			poisson_impulse_times[get_random(T_sim_cycle)] = 1;
//		}
//
//		int num = 0;
//		for (int t = 0; t < T_sim_cycle; t++){
//			if (poisson_impulse_times[t]){
//				poisson_times[Nneur*num + n] = current_time + 1 + t;
//				num++;
//			}
//		}
//		memset(poisson_impulse_times, 0, T_sim_cycle*sizeof(int));
//	}
//	memset(poisson_processed, 0, Nneur*sizeof(int));
//	free(poisson_impulse_times);
//}

void malloc_neur_memory(){
	V_ms = new float[Nneur];
	V_ms_last = new float[Nneur];
	m_chs = new float[Nneur];
	n_chs = new float[Nneur];
	h_chs = new float[Nneur];
	I_syns = new float[Nneur];
	I_syns_last = new float[Nneur];
	I_es = new float[Nneur];
	I_psns = new float[Nneur];
	y_psns = new float[Nneur];
	psn_times = new int[Nneur*T_sim/T_sim_part_psn];
	num_psns = new int[Nneur];
	// if num-th spike occur at a time t on n-th neuron then,
	// t is stored in element with index Nneur*num + n
	// spike_times[Nneur*num + n] = t
	spike_times = new int[Nneur*T_sim/T_sim_ratio];
	num_spikes_neur = new int[Nneur];
	memset(I_syns, 0, Nneur*sizeof(float));
	memset(I_syns_last, 0, Nneur*sizeof(float));
	memset(I_psns, 0, Nneur*sizeof(float));
	memset(y_psns, 0, Nneur*sizeof(float));
	memset(num_spikes_neur, 0, Nneur*sizeof(int));
	memset(num_psns, 0, Nneur*sizeof(int));
}

void malloc_conn_memory(){
	ys = new float[Ncon];
	I_pscs = new float[Ncon];
	weights = new float[Ncon];
	pre_conns = new int[Ncon];
	post_conns = new int[Ncon];
	delays = new int[Ncon];
	num_spikes_syn = new int[Ncon];
	memset(ys, 0, Ncon*sizeof(int));
	memset(I_pscs, 0, Ncon*sizeof(int));
	memset(num_spikes_syn, 0, Ncon*sizeof(int));
}

void copy2device(){
	int n_fsize = Nneur*sizeof(float);
	int n_isize = Nneur*sizeof(int);
	cudaMalloc((void**) &V_ms_dev, n_fsize);
	cudaMalloc((void**) &V_ms_last_dev, n_fsize);
	cudaMalloc((void**) &m_chs_dev, n_fsize);
	cudaMalloc((void**) &n_chs_dev, n_fsize);
	cudaMalloc((void**) &h_chs_dev, n_fsize);
	cudaMalloc((void**) &I_syns_dev, n_fsize);
	cudaMalloc((void**) &I_syns_last_dev, n_fsize);
	cudaMalloc((void**) &I_es_dev, n_fsize);
	cudaMalloc((void**) &I_psns_dev, n_fsize);
	cudaMalloc((void**) &y_psns_dev, n_fsize);
	cudaMalloc((void**) &psn_times_dev, n_isize*T_sim/T_sim_part_psn);
	cudaMalloc((void**) &num_psns_dev, n_isize);

	cudaMalloc((void**) &spike_times_dev, n_isize*T_sim/T_sim_ratio);
	cudaMalloc((void**) &num_spikes_neur_dev, n_isize);

	int s_fsize = Ncon*sizeof(float);
	int s_isize = Ncon*sizeof(int);
	cudaMalloc((void**) &ys_dev, s_fsize);
	cudaMalloc((void**) &I_pscs_dev, s_fsize);
	cudaMalloc((void**) &weights_dev, s_fsize);
	cudaMalloc((void**) &pre_conns_dev, s_isize);
	cudaMalloc((void**) &post_conns_dev, s_isize);
	cudaMalloc((void**) &delays_dev, s_isize);
	cudaMalloc((void**) &num_spikes_syn_dev, s_isize);

	cudaMemcpy(V_ms_dev, V_ms, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(V_ms_last_dev, V_ms_last, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(m_chs_dev, m_chs, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(n_chs_dev, n_chs, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(h_chs_dev, h_chs, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(I_syns_dev, I_syns, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(I_syns_last_dev, I_syns_last, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(I_es_dev, I_es, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(I_psns_dev, I_psns, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(y_psns_dev, y_psns, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(psn_times_dev, psn_times, n_isize*T_sim/T_sim_part_psn, cudaMemcpyHostToDevice);
	cudaMemcpy(num_psns_dev, num_psns, n_isize, cudaMemcpyHostToDevice);

	cudaMemcpy(spike_times_dev, spike_times, n_isize*T_sim/T_sim_ratio, cudaMemcpyHostToDevice);
	cudaMemcpy(num_spikes_neur_dev, num_spikes_neur, n_isize, cudaMemcpyHostToDevice);

	cudaMemcpy(ys_dev, ys, s_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(I_pscs_dev, I_pscs, s_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(weights_dev, weights, s_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(pre_conns_dev, pre_conns, s_isize, cudaMemcpyHostToDevice);
	cudaMemcpy(post_conns_dev, post_conns, s_isize, cudaMemcpyHostToDevice);
	cudaMemcpy(delays_dev, delays, s_isize, cudaMemcpyHostToDevice);
	cudaMemcpy(num_spikes_syn_dev, num_spikes_syn, s_isize, cudaMemcpyHostToDevice);
}
