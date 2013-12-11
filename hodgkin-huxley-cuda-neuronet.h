/*
 * hodgkin-huxley-neuronet-cuda.h
 *
 *  Created on: 09.12.2013
 *      Author: postdoc3
 */
#include <cstring>

float* V_ms;
float* V_ms_last;
float* m_chs;
float* n_chs;
float* h_chs;
float* I_syns;
float* I_syns_; // full postsynaptic current on each neuron
float* I_es;
float* I_poisson;
float* y_poisson;

int* spike_times; // spike times for each neuron
int* num_spikes;  // numver of spikes on eash neuron

float* ys;
float* I_psns; // partial postsynaptic current on each synapse
float* weights;
int* pre_conns;
int* post_conns;
int* delays; // delays in integration steps
int* num_spk_proc; // number of processed spikes by each synapse

int Nneur = 10;
int Ncon;
int T_sim;
int spike_arr_dim;

float V_m, n_ch, m_ch, h_ch;
float v1, v2, v3, v4;
float n1, n2, n3, n4;
float m1, m2, m3, m4;
float h1, h2, h3, h4;

void malloc_neur_memory(){
	V_ms = new float[Nneur];
	V_ms_last = new float[Nneur];
	m_chs = new float[Nneur];
	n_chs = new float[Nneur];
	h_chs = new float[Nneur];
	I_syns = new float[Nneur];
	I_syns_ = new float[Nneur];
	I_es = new float[Nneur];
	I_poisson = new float[Nneur];
	y_poisson = new float[Nneur];
	// if num-th spike occur at a time t on n-th neuron then,
	// t is stored in element with index T_sim/5*num + n
	// spike_times[T_sim/5*num + n] = t
	spike_times = new int[Nneur*T_sim/5];
	num_spikes = new int[Nneur];
	memset(I_syns, 0, Nneur*sizeof(float));
	memset(I_syns_, 0, Nneur*sizeof(float));
	memset(I_poisson, 0, Nneur*sizeof(float));
	memset(y_poisson, 0, Nneur*sizeof(float));
	memset(num_spikes, 0, Nneur*sizeof(int));
}

void malloc_conn_memory(){
	ys = new float[Ncon];
	I_psns = new float[Ncon];
	weights = new float[Ncon];
	pre_conns = new int[Ncon];
	post_conns = new int[Ncon];
	delays = new int[Ncon];
	num_spk_proc = new int[Ncon];
	memset(ys, 0, Ncon*sizeof(int));
	memset(I_psns, 0, Ncon*sizeof(int));
	memset(num_spk_proc, 0, Ncon*sizeof(int));
}

void init_conns_from_file();

void init_neurs_from_file();
