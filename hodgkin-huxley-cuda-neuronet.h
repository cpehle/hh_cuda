/*
 * hodgkin-huxley-neuronet-cuda.h
 *
 *  Created on: 09.12.2013
 *      Author: postdoc3
 */
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

float* V_ms;
float* V_ms_last;
float* m_chs;
float* n_chs;
float* h_chs;
float* I_syns;
float* I_syns_last; // full postsynaptic current on each neuron
float* I_es;
float* I_psns;
float* y_psns;
int* psn_times;
int* num_psns;

float* V_ms_dev;
float* V_ms_last_dev;
float* m_chs_dev;
float* n_chs_dev;
float* h_chs_dev;
float* I_syns_dev;
float* I_syns_last_dev;
float* I_es_dev;
float* I_psns_dev;
float* y_psns_dev;
int* psn_times_dev;
int* num_psns_dev;

int* spike_times; // spike times for each neuron
int* num_spikes_neur;  // numver of spikes on eash neuron

int* spike_times_dev;
int* num_spikes_neur_dev;

float* ys;
float* I_pscs; // partial postsynaptic current on each synapse
float* weights;
int* pre_conns;
int* post_conns;
int* delays; // delays in integration steps
int* num_spikes_syn; // number of processed spikes by each synapse

float* ys_dev;
float* I_pscs_dev;
float* weights_dev;
int* pre_conns_dev;
int* post_conns_dev;
int* delays_dev;
int* num_spikes_syn_dev;

int Nneur = 1024;
int Ncon;

int T_sim;
// maximum part of simulating time for which is allocated memory
// so the maximum number of spikes per neuron which can be processed is
// T_sim/T_sim_part
int T_sim_part = 100;
// similarly to poisson spikes for each neuron
// defined by poisson frequence
// T_sim_part_poisson = 1000/(h*max_rate)
int T_sim_part_psn = 40;

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
	spike_times = new int[Nneur*T_sim/T_sim_part];
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

	cudaMalloc((void**) &spike_times_dev, n_isize*T_sim/T_sim_part);
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

	cudaMemcpy(spike_times_dev, spike_times, n_isize*T_sim/T_sim_part, cudaMemcpyHostToDevice);
	cudaMemcpy(num_spikes_neur_dev, num_spikes_neur, n_isize, cudaMemcpyHostToDevice);

	cudaMemcpy(ys_dev, ys, s_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(I_pscs_dev, I_pscs, s_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(weights_dev, weights, s_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(pre_conns_dev, pre_conns, s_isize, cudaMemcpyHostToDevice);
	cudaMemcpy(post_conns_dev, post_conns, s_isize, cudaMemcpyHostToDevice);
	cudaMemcpy(delays_dev, delays, s_isize, cudaMemcpyHostToDevice);
	cudaMemcpy(num_spikes_syn_dev, num_spikes_syn, s_isize, cudaMemcpyHostToDevice);
}


int get_random(int max){

	if (RAND_MAX == 32767){
		return ((RAND_MAX + 1)*(long)rand() + rand()) % max;
	} else {
		return rand() % max;
	}
}

void init_conns_from_file();

void init_neurs_from_file();

void save2file();
