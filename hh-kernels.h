/*
 * hh-kernels.h
 *
 *  Created on: Aug 29, 2015
 *      Author: pavel
 */

#ifndef HH_KERNELS_H_
#define HH_KERNELS_H_

#include <curand_kernel.h>

void integrate_neurons(dim3 NumBlocks, dim3 BlockSz,
                        float* V_m, float* V_m_last, float* n_ch, float* m_ch, float* h_ch,
                        int* spike_time, int* num_spike_neur,
                        float* I_e, float* y, float* I_syn, float* y_psn, float* I_psn, unsigned int* psn_time, unsigned int* psn_seed,
                        float* exp_w_p, float exp_psc, float rate,
                        int Nneur, int t, float h, float* D, float* Inoise, curandState* state, float* Vrec);

void init_poisson(dim3 NumBlocks, dim3 BlockSz,
                            unsigned int* psn_time, unsigned int *psn_seed, unsigned int seed, float rate, float h, int Nneur, int BundleSize);

void init_noise(dim3 NumBlocks, dim3 BlockSz,
        curandState* state, float* Inoise, float* D, unsigned int seed, int Nneur, int BundleSize);

void integrate_synapses(dim3 NumBlocks, dim3 BlockSz,
        float* y, float* weight, int* delay, int* pre_conn, int* post_conn,
        int* spike_time, unsigned int* num_spike_syn, int* num_spike_neur, int t, int Nneur, int Ncon);

#endif /* HH_KERNELS_H_ */

