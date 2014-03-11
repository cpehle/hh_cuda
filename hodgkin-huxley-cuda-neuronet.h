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

// Variables for each neuron
float* V_ms;
float* V_ms_last;
float* m_chs;
float* n_chs;
float* h_chs;
float* I_es;

float* ys;       // neurotransmitter concentration on each neuron (without poisson)
float* I_syns;   // input synaptic current
float* y_psns;   // neurotransmitter concentration induced by poisson noise
float* I_psns;   // input current induced by poisson

float* I_last;   // previous time step current value

float *exp_w_p;

int* spike_times; // spike times for each neuron
int* num_spikes_neur;  // number of spikes on each neuron

float* V_ms_dev;
float* V_ms_last_dev;
float* m_chs_dev;
float* n_chs_dev;
float* h_chs_dev;
float* I_es_dev;

float* ys_dev;
float* I_syns_dev;
float* y_psns_dev;
float* I_psns_dev;

float* I_last_dev;

float *exp_w_p_dev;

int* spike_times_dev;
int* num_spikes_neur_dev;

int* psn_times_dev;
unsigned int* psn_seeds_dev;

// Variables for each synapse
float* weights;
int* pre_conns;
int* post_conns;
int* delays; // delays in integration steps
int* num_spikes_syn; // number of processed spikes by each synapse

float* weights_dev;
int* pre_conns_dev;
int* post_conns_dev;
int* delays_dev;
int* num_spikes_syn_dev;

int Ncon;

int T_sim;

void init_conns_from_file();

void init_neurs_from_file();

void save2file();

void swap_spikes();

void malloc_neur_memory();

void malloc_conn_memory();

void copy2device();

void init_poisson();

void clear_files();

void init_params(int, char*[]);
