/*
 * hh-kernels.cu
 *
 *  Created on: Aug 29, 2015
 *      Author: pavel
 */
#include <curand_kernel.h>

#define recInt 5
#define T_sim_partial 10000 // in time frames
#define Nrec 10

// neuron parameters
#define Cm_    1.0f //  inverse of membrane capacity, 1/pF
#define g_Na  120.0f // nS
#define g_K   36.0f
#define g_L   .3f
#define E_K   -77.0f
#define E_Na  55.0f
#define E_L   -54.4f
#define V_peak 25.0f

#define tau_cor 2.0f

__device__ float get_random(unsigned int *seed){
    // return random number homogeneously distributed in interval [0:1]
    unsigned long a = 16807;
    unsigned long m = 2147483647;
    unsigned long x = (unsigned long) *seed;
    x = (a * x) % m;
    *seed = (unsigned int) x;
    return ((float)x)/m;
}

__device__ float hh_Vm(float V, float n_ch, float m_ch, float h_ch, float I_syn, float I_e, float h){
    return (I_e - g_K*(V - E_K)*n_ch*n_ch*n_ch*n_ch - g_Na*(V - E_Na)*m_ch*m_ch*m_ch*h_ch - g_L*(V - E_L) + I_syn)*h*Cm_;
}

__device__ float hh_n_ch(float V, float n_ch, float h){
    float temp = 1.0f - expf(-(V + 55.0f)*0.1f);
    if (temp != 0.0f){
        return (.01f*(1.0f - n_ch)*(V + 55.0f)/temp - 0.125*n_ch*expf(-(V + 65.0f)*0.0125f))*h;
    } else {
//      printf("Деление на ноль, n! \n");
//      to understand why it'so, calculate the limit for v/(1 - exp(v/10)) then v tend to 0
        return (0.1f*(1.0f - n_ch)- 0.125*n_ch*expf(-(V + 65.0f)*0.0125f))*h;
    }
}

__device__ float hh_m_ch(float V, float m_ch, float h){
    float temp = 1.0f - expf(-(V + 40.0f)*0.1f);
    if (temp != 0.0f){
        return (0.1f*(1.0f - m_ch)*(V + 40.0f)/temp - 4.0f*m_ch*expf(-(V + 65.0f)*0.055555556f))*h;
    } else {
//      printf("Деление на ноль, m! \n");
        return ((1.0f - m_ch) - 4.0f*m_ch*expf(-(V + 65.0f)*0.055555556f))*h;
    }
}

__device__ float hh_h_ch(float V, float h_ch, float h){
    return (.07f*(1.0f - h_ch)*expf(-(V + 65.0f)*0.05f) - h_ch/(1.0f + expf(-(V + 35.0f)*0.1f)))*h;
}

__global__ void gpu_init_poisson(unsigned int* psn_time, unsigned int *psn_seed, unsigned int seed, float rate, float h, int Nneur, int BundleSize){
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int neur = n % BundleSize;
    if (n < Nneur){
        psn_seed[n] = 100000*(seed + neur + 1);
        psn_time[n] = 1 -(1000.0f/(h*rate))*logf(get_random(psn_seed + n));
    }
}

__global__ void gpu_init_noise(curandState* state, float* Inoise, float* D, unsigned int seed, int Nneur, int BundleSize){
    int n = blockIdx.x*blockDim.x + threadIdx.x;
    int neur = n % BundleSize;
    if (n < Nneur){
        curand_init(0, neur + seed, 0, &state[n]);
//      Inoise[n] += sqrt(D[n]/tau_cor)*curand_normal(&state[n]);
    }
}

__global__ void gpu_integrate_synapses(float* y, float* weight, int* delay, int* pre_conn, int* post_conn,
        int* spike_time, unsigned int* num_spike_syn, int* num_spike_neur, int t, int Nneur, int Ncon){
    int s = blockDim.x*blockIdx.x + threadIdx.x;
    if (s < Ncon){
        int pre_neur = pre_conn[s];
        // if we processed less spikes than there is in presynaptic neuron
        // we need to check whether new spikes at arrive this moment of time
        if (num_spike_syn[s] < num_spike_neur[pre_neur]){
            if (spike_time[Nneur*num_spike_syn[s] + pre_neur] == t - delay[s]){
                atomicAdd(&y[post_conn[s]], weight[s]);
                num_spike_syn[s]++;
            }
        }
    }
}

__global__ void gpu_integrate_neurons(
        float* V_m, float* V_m_last, float* n_ch, float* m_ch, float* h_ch,
        int* spike_time, int* num_spike_neur,
        float* I_e, float* y, float* I_syn, float* y_psn, float* I_psn, unsigned int* psn_time, unsigned int* psn_seed,
        float* exp_w_p, float exp_psc, float rate,
        int Nneur, int t, float h, float* D, float* Inoise, curandState* state, float* Vrec){
        int n = blockIdx.x*blockDim.x + threadIdx.x;
        if (n < Nneur){
            float I_syn_last = I_psn[n] + I_syn[n];
            I_psn[n]  = (y_psn[n]*h + I_psn[n])*exp_psc;
            y_psn[n] *= exp_psc;

            I_syn[n]  = (y[n]*h + I_syn[n])*exp_psc;
            y[n] *= exp_psc;

            // if where is poisson impulse on neuron
            if (psn_time[n] == t){
                y_psn[n] += exp_w_p[n];
                psn_time[n] += 1 + (unsigned int) (-(1000.0f/(rate*h))*logf(get_random(psn_seed + n)));
//              psn_time[n] += 1 + (unsigned int) (-(1000.0f/(rate*h))*logf(curand_uniform(&state[n])));
            }
            float V_mem, n_channel, m_channel, h_channel;
            float v1, v2, v3, v4;
            float n1, n2, n3, n4;
            float m1, m2, m3, m4;
            float h1, h2, h3, h4;
            float Inoise_;
            float ns1, ns2, ns3, ns4;

            float dNoise = 0.0f;
//          float dNoise = sqrtf(2.0f*h*D[n])*curand_normal(&state[n]);
//          Inoise[n] = sqrtf(h*D[n])*curand_normal(&state[n])/h;

            V_mem = V_m[n];
            n_channel = n_ch[n];
            m_channel = m_ch[n];
            h_channel = h_ch[n];
            Inoise_ = Inoise[n];
            v1 = hh_Vm(V_m[n], n_ch[n], m_ch[n], h_ch[n], I_syn_last + Inoise[n], I_e[n], h);
            n1 = hh_n_ch(V_m[n], n_ch[n], h);
            m1 = hh_m_ch(V_m[n], m_ch[n], h);
            h1 = hh_h_ch(V_m[n], h_ch[n], h);
            ns1 = (-Inoise[n]*h + dNoise)/tau_cor;
            V_m[n] = V_mem + v1/2.0f;
            n_ch[n] = n_channel + n1/2.0f;
            m_ch[n] = m_channel + m1/2.0f;
            h_ch[n] = h_channel + h1/2.0f;
            Inoise[n] = Inoise_ + ns1/2.0f;

            v2 = hh_Vm(V_m[n], n_ch[n], m_ch[n], h_ch[n], (I_syn[n] + I_psn[n] + I_syn_last)/2.0f + Inoise[n] , I_e[n], h);
            n2 = hh_n_ch(V_m[n], n_ch[n], h);
            m2 = hh_m_ch(V_m[n], m_ch[n], h);
            h2 = hh_h_ch(V_m[n], h_ch[n], h);
            ns2 = (-Inoise[n]*h + dNoise)/tau_cor;
            V_m[n] = V_mem + v2/2.0f;
            n_ch[n] = n_channel + n2/2.0f;
            m_ch[n] = m_channel + m2/2.0f;
            h_ch[n] = h_channel + h2/2.0f;
            Inoise[n] = Inoise_ + ns2/2.0f;


            v3 = hh_Vm(V_m[n], n_ch[n], m_ch[n], h_ch[n], (I_syn[n] + I_psn[n] + I_syn_last)/2.0f + Inoise[n], I_e[n], h);
            n3 = hh_n_ch(V_m[n], n_ch[n], h);
            m3 = hh_m_ch(V_m[n], m_ch[n], h);
            h3 = hh_h_ch(V_m[n], h_ch[n], h);
            ns3 = (-Inoise[n]*h + dNoise)/tau_cor;
            V_m[n] = V_mem + v3;
            n_ch[n] = n_channel + n3;
            m_ch[n] = m_channel + m3;
            h_ch[n] = h_channel + h3;
            Inoise[n] = Inoise_ + ns3;


            v4 = hh_Vm(V_m[n], n_ch[n], m_ch[n], h_ch[n], I_syn[n] + I_psn[n] + Inoise[n], I_e[n], h);
            n4 = hh_n_ch(V_m[n], n_ch[n], h);
            m4 = hh_m_ch(V_m[n], m_ch[n], h);
            h4 = hh_h_ch(V_m[n], h_ch[n], h);
            ns4 = (-Inoise[n]*h + dNoise)/tau_cor;

            V_m[n] = V_mem + (v1 + 2.0f*v2 + 2.0f*v3 + v4)/6.0f;
            n_ch[n] = n_channel + (n1 + 2.0f*n2 + 2.0f*n3 + n4)/6.0f;
            m_ch[n] = m_channel + (m1 + 2.0f*m2 + 2.0f*m3 + m4)/6.0f;
            h_ch[n] = h_channel + (h1 + 2.0f*h2 + 2.0f*h3 + h4)/6.0f;
            Inoise[n] = Inoise_ + (ns1 + 2.0f*(ns2 + ns3) + ns4)/6.0f;

            // checking if there's spike on neuron
            if (V_m[n] > V_peak && V_mem > V_m[n] && V_m_last[n] <= V_mem){
                if (num_spike_neur[n] == 0 || t - spike_time[Nneur*(num_spike_neur[n] - 1) + n] > 5.0f/h){
                    spike_time[Nneur*num_spike_neur[n] + n] = t;
                    num_spike_neur[n]++;
                }
            }
            V_m_last[n] = V_mem;
#ifdef OSCILL_SAVE
            if (t % recInt == 0){
                Vrec[Nrec*(t % T_sim_partial/recInt) + n] = V_m[n+50];
            }
#endif
        }
}

__host__ void integrate_neurons(dim3 NumBlocks, dim3 BlockSz,
                        float* V_m, float* V_m_last, float* n_ch, float* m_ch, float* h_ch,
                        int* spike_time, int* num_spike_neur,
                        float* I_e, float* y, float* I_syn, float* y_psn, float* I_psn, unsigned int* psn_time, unsigned int* psn_seed,
                        float* exp_w_p, float exp_psc, float rate,
                        int Nneur, int t, float h, float* D, float* Inoise, curandState* state, float* Vrec){

    gpu_integrate_neurons<<<NumBlocks, BlockSz>>>(V_m, V_m_last, n_ch, m_ch, h_ch,
                                        spike_time, num_spike_neur,
                                        I_e, y, I_syn, y_psn, I_psn, psn_time, psn_seed,
                                        exp_w_p, exp_psc, rate,
                                        Nneur, t, h, D, Inoise, state, Vrec);

}

__host__ void init_poisson(dim3 NumBlocks, dim3 BlockSz,
                            unsigned int* psn_time, unsigned int *psn_seed, unsigned int seed, float rate, float h, int Nneur, int BundleSize){

    gpu_init_poisson<<<NumBlocks, BlockSz>>>(psn_time, psn_seed, seed, rate, h, Nneur, BundleSize);
}

__host__ void init_noise(dim3 NumBlocks, dim3 BlockSz,
        curandState* state, float* Inoise, float* D, unsigned int seed, int Nneur, int BundleSize){

    gpu_init_noise<<<NumBlocks, BlockSz>>>(state,  Inoise,  D, seed, Nneur, BundleSize);
}

__host__ void integrate_synapses(dim3 NumBlocks, dim3 BlockSz,
        float* y, float* weight, int* delay, int* pre_conn, int* post_conn,
        int* spike_time, unsigned int* num_spike_syn, int* num_spike_neur, int t, int Nneur, int Ncon){

    gpu_integrate_synapses<<<NumBlocks, BlockSz>>>(y,  weight, delay, pre_conn, post_conn,
            spike_time, num_spike_syn, num_spike_neur, t, Nneur, Ncon);
}
