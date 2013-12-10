/*
 * hodgkin-huxley-neuronet-cuda.h
 *
 *  Created on: 09.12.2013
 *      Author: postdoc3
 */

float* V_ms;
float* m_chs;
float* n_chs;
float* h_chs;
float* I_syns;
float* I_syns_; // full postsynaptic current on each neuron
float* I_es;
float* I_poisson;
float* y_poisson;

float* ys;
float* I_psns; // partial postsynaptic current on each synapse
int* pre_conns;
int* post_conns;
int* delays; // delays in integration steps

int Nneur = 2;
int Ncon = 1;

void malloc_neur_memory(){
	V_ms = new float[Nneur];
	m_chs = new float[Nneur];
	n_chs = new float[Nneur];
	h_chs = new float[Nneur];
	I_syns = new float[Nneur];
	I_syns_ = new float[Nneur];
	I_es = new float[Nneur];
	I_poisson = new float[Nneur];
	y_poisson = new float[Nneur];
}

void malloc_conn_memory(){
	ys = new float[Ncon];
	I_psns = new float[Ncon];
	pre_conns = new int[Ncon];
	post_conns = new int[Ncon];
	delays = new int[Ncon];
}

void init_conns_from_file(){
	ys[0] = 0.0f;
	I_psns[0] = 0.0f;
	pre_conns[0] = 0;
	post_conns[0] = 1;
	delays[0] = 0.0f;
}

void ini_neurs_from_file(){
	V_ms[0] = 32.906693f;
	V_ms[1] = 32.906693f;

	m_chs[0] = 0.913177f;
	n_chs[0] = 0.574678f;
	h_chs[0] = 0.223994f;

	m_chs[1] = 0.913177f;
	n_chs[1] = 0.574678f;
	h_chs[1] = 0.223994f;

	I_syns[0] = 0.0f;
	I_syns[1] = 0.0f;
	I_syns_[0] = 0.0f;
	I_syns_[1] = 0.0f;
	I_es[0] = 5.27f;
	I_es[1] = 0.0f;
}
