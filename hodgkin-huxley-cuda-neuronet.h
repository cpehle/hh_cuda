/*
 * hodgkin-huxley-neuronet-cuda.h
 *
 *  Created on: 09.12.2013
 *      Author: postdoc3
 */
#include <cstring>
//#include <cuda.h>
//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
#include <iostream>
#include "hh-kernels.h"

#define CUDA_CHECK_RETURN(value) {										\
	cudaError_t _m_cudaStat = value;									\
	if (_m_cudaStat != cudaSuccess) {									\
		fprintf(stderr, "Error %s at line %d in file %s\n",				\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);	\
		exit(1);														\
	} }

#define NEUR_BLOCK_SIZE 128
#define SYN_BLOCK_SIZE 512

#define recInt 5
#define T_sim_partial 10000 // in time frames
#define Nrec 10

unsigned int time_part_syn;
// maximum part of simulating time for which is allocated memory
// time_part_syn <= T[ms]/h[ms]
// 15.0f is rough period
// so the maximum number of spikes per neuron which can be processed is
// T_sim_particular/time_part_syn

// interval of saving results to file,
// if greater then fragmentation becomes less,
// but more RAM is used
// in frames
unsigned int SaveIntervalTIdx = 100000;

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

unsigned int* psn_times;
unsigned int* psn_seeds;

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

float *exp_w_p_dev;

int* spike_times_dev;
int* num_spikes_neur_dev;

unsigned int* psn_times_dev;
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
unsigned int* num_spikes_syn; // number of processed spikes by each synapse

float* weights_dev;
int* pre_conns_dev;
int* post_conns_dev;
int* delays_dev;
unsigned int* num_spikes_syn_dev;

float* res_times;
int* res_senders;
int* num_spk_in_bund;

float* Vrec;
float* Vrec_dev;

unsigned int Ncon;

unsigned int T_sim;

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

void saveIVP2Fl();

void array2file(char* fl_name, unsigned int N, unsigned int arr[]);

void file2array(char* fl_name, unsigned int N, unsigned int arr[]);

void raw2file(char* fl_name, unsigned int N, float arr[]);

void raw2array(char* fl_name, unsigned int N, float arr[]);

void save_oscill(int tm, bool lastFlag=false);

void init_params(int, char*[]);

void apndResToFile();

void clear_oscill_file();
