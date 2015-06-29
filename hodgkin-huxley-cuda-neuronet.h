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
#include <curand_kernel.h>
#include <iostream>

#define CUDA_CHECK_RETURN(value) {							\
	cudaError_t _m_cudaStat = value;						\
	if (_m_cudaStat != cudaSuccess) {						\
		fprintf(stderr, "Error %s at line %d in file %s\n",			\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);	\
		exit(1);								\
	} }

#define NEUR_BLOCK_SIZE 128
#define SYN_BLOCK_SIZE 512

// neuron parameters
__constant__ float Cm    = 1.0f; //  inverse of membrane capacity, 1/pF
__constant__ float g_Na  = 120.0f; // nS
__constant__ float g_K   = 36.0f;
__constant__ float g_L   = .3f;
__constant__ float E_K   = -77.0f;

// @TODO для нейронов в бистабильном режиме E_Na = 50.
__constant__ float E_Na  = 55.0f;
//__constant__ float E_Na  = 50.0f;

__constant__ float E_L   = -54.4f;
__constant__ float V_peak = 18.0f;
__constant__ float tau_cor = 2.0f;
__constant__ int recInt_dev = 10;
int recInt = 10;


int T_sim_partial = 100000; // in time frames
__constant__ unsigned int T_sim_part_dev = 100000; // in time frames

int time_part_syn;
// maximum part of simulating time for which is allocated memory
// time_part_syn <= T[ms]/h[ms]
// 15.0f is rough period
// so the maximum number of spikes per neuron which can be processed is
// T_sim_particular/time_part_syn

// interval of saving results to file,
// if greater then fragmentation becomes less,
// but more RAM is used
// in frames
int SaveIntervalTIdx = 1000000;

float tau_psc = 0.2f;
float exp_psc;

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

float* Ds_dev;
float* Ds_host;
float* Inoise_dev;
curandState* noise_states_dev;
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

float* res_times;
int* res_senders;
int* num_spk_in_bund;

float* Vrec;
float* Vrec_dev;

int Ncon;

int T_sim;

int gaussNoiseFlag = 0;

void init_conns_from_file();

void init_neurs_from_file();

void save2HOST();

void swap_spikes();

void clearResFiles();

void malloc_neur_memory();

void malloc_conn_memory();

void copy2device();

void init_poisson();

void clear_files();

void save_oscill(int tm, bool lastFlag=false);

void init_params(int, char*[]);

void apndResToFile();

void clear_oscill_file();
