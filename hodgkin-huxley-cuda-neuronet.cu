#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <climits>
#include <ctime>
#ifdef WITH_MPI
#include <mpi.h>
#endif
#include "hodgkin-huxley-cuda-neuronet.h"

unsigned int seed = 1;
float h = 0.1f;
float SimulationTime = 10000.0f; // in ms

unsigned int Nneur = 2;
unsigned int W_P_NUM_BUND = 1; // number of different poisson weights
unsigned int W_P_BUND_SZ = Nneur/W_P_NUM_BUND; // Number of neurons in bundle with same w_ps
unsigned int BUND_SZ = 2;  // Number of neurons in a single realization
unsigned int NUM_BUND = W_P_BUND_SZ/BUND_SZ;

// connection parameters
float I_e = 5.27f;
float dI_e = 0.0f;
float w_p_start = 1.8f; // pA
float w_p_stop = 2.0f;
float w_n = 5.4f;
float rate = 200.0f;

int One = 0;

char f_name[500] = "0";
char par_f_name[500] = "nn_params_2.csv";
char ivp_fname[500] = "";

unsigned int Tstart = 1;

__device__ __host__ float get_random(unsigned int *seed){
	// return random number homogeneously distributed in interval [0:1]
	unsigned long a = 16807;
	unsigned long m = 2147483647;
	unsigned long x = (unsigned long) *seed;
	x = (a * x) % m;
	*seed = (unsigned int) x;
	return ((float)x)/m;
}

__device__ float hh_Vm(float V, float n_ch, float m_ch, float h_ch, float I_syn, float I_e, float h){
	return (I_e - g_K*(V - E_K)*n_ch*n_ch*n_ch*n_ch - g_Na*(V - E_Na)*m_ch*m_ch*m_ch*h_ch - g_L*(V - E_L) + I_syn)*h*Cm;
}

__device__ float hh_n_ch(float V, float n_ch, float h){
	float temp = 1.0f - expf(-(V + 55.0f)*0.1f);
	if (temp != 0.0f){
		return (.01f*(1.0f - n_ch)*(V + 55.0f)/temp - 0.125*n_ch*expf(-(V + 65.0f)*0.0125f))*h;
	} else {
//		printf("Деление на ноль, n! \n");
//      to understand why it'so, calculate the limit for v/(1 - exp(v/10)) then v tend to 0
		return (0.1f*(1.0f - n_ch)- 0.125*n_ch*expf(-(V + 65.0f)*0.0125f))*h;
	}
}

__device__ float hh_m_ch(float V, float m_ch, float h){
	float temp = 1.0f - expf(-(V + 40.0f)*0.1f);
	if (temp != 0.0f){
		return (0.1f*(1.0f - m_ch)*(V + 40.0f)/temp - 4.0f*m_ch*expf(-(V + 65.0f)*0.055555556f))*h;
	} else {
//		printf("Деление на ноль, m! \n");
		return ((1.0f - m_ch) - 4.0f*m_ch*expf(-(V + 65.0f)*0.055555556f))*h;
	}
}

__device__ float hh_h_ch(float V, float h_ch, float h){
	return (.07f*(1.0f - h_ch)*expf(-(V + 65.0f)*0.05f) - h_ch/(1.0f + expf(-(V + 35.0f)*0.1f)))*h;
}

__global__ void init_poisson(unsigned int* psn_time, unsigned int *psn_seed, unsigned int seed, float rate, float h, int Nneur, int BundleSize){
	int n = blockIdx.x*blockDim.x + threadIdx.x;
	int neur = n % BundleSize;
	if (n < Nneur){
		psn_seed[n] = 100000*(seed + neur + 1);
		psn_time[n] = -(1000.0f/(h*rate))*logf(get_random(psn_seed + n));
	}
}

__global__ void init_noise(curandState* state, float* Inoise, float* D, unsigned int seed, int Nneur, int BundleSize){
	int n = blockIdx.x*blockDim.x + threadIdx.x;
	int neur = n % BundleSize;
	if (n < Nneur){
		curand_init(0, neur + seed, 0, &state[n]);
		Inoise[n] += sqrt(D[n]/tau_cor)*curand_normal(&state[n]);
	}
}

__global__ void integrate_synapses(float* y, float* weight, int* delay, int* pre_conn, int* post_conn,
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

__global__ void integrate_neurons(
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
			}
			float V_mem, n_channel, m_channel, h_channel;
			float v1, v2, v3, v4;
			float n1, n2, n3, n4;
			float m1, m2, m3, m4;
			float h1, h2, h3, h4;
			float Inoise_;
			float ns1, ns2, ns3, ns4;

			float dNoise = sqrtf(2.0f*h*D[n])*curand_normal(&state[n]);
//			Inoise[n] = sqrtf(h*D[n])*curand_normal(&state[n])/h;

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

//			V_m[n] = V_mem + v1;
//			n_ch[n] = n_channel + n1;
//			m_ch[n] = m_channel + m1;
//			h_ch[n] = h_channel + h1;
//			Inoise[n] = Inoise_ + ns1;

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
				Vrec[Nneur*(t % T_sim_partial/recInt) + n] = V_m[n];
			}
#endif
		}
}

using namespace std;

int world_size, world_rank;

int main(int argc, char* argv[]){
    init_params(argc, argv);

    #ifdef WITH_MPI
//    cout << "Calculating with MPI" << endl;
    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    switch (deviceCount){
        // @TODO hardcode! 3 - number of GPU devices on each node of Lobachevsky Supercomputer
        case 3: cudaSetDevice(world_rank % 3); break;
        case 1: cudaSetDevice(0); break;
        default: exit(EXIT_FAILURE);
    }
    I_e += dI_e*world_rank;
#else
    cudaSetDevice(0);
#endif
    sprintf(f_name, "%s/N_%i_rate_%.1f_w_n_%.1f_Ie_%.2f", f_name, BUND_SZ, rate, w_n, I_e);

    exp_psc = expf(-h/tau_psc);
	time_part_syn = 10.0f/h;
	T_sim = SimulationTime/h;
	init_neurs_from_file();
	init_conns_from_file();
	copy2device();
	clearResFiles();
#ifdef OSCILL_SAVE
	clear_oscill_file();
#endif
	if (strcmp(ivp_fname, "") == 0){
        init_poisson<<<dim3(Nneur/NEUR_BLOCK_SIZE + 1), dim3(NEUR_BLOCK_SIZE)>>>(psn_times_dev, psn_seeds_dev, seed, rate, h, Nneur, W_P_BUND_SZ);
	}
	init_noise<<<dim3(Nneur/NEUR_BLOCK_SIZE + 1), dim3(NEUR_BLOCK_SIZE)>>>(noise_states_dev, Inoise_dev, Ds_dev, seed, Nneur, W_P_BUND_SZ);

	time_t curr_time = time(0);
    char* st = asctime(localtime(&curr_time));
	cout << "Start: for rank: " << world_rank << " " << st << endl;
    for (unsigned int t = Tstart; t < T_sim; t++){
#ifdef OSCILL_SAVE
		cudaDeviceSynchronize();
    	if (t % T_sim_partial == 0){
			CUDA_CHECK_RETURN(cudaMemcpy(Vrec, Vrec_dev, Nneur*T_sim_partial/recInt*sizeof(float), cudaMemcpyDeviceToHost));
			save_oscill(t);
    	}
#endif
		integrate_neurons<<<dim3((Nneur + NEUR_BLOCK_SIZE - 1)/NEUR_BLOCK_SIZE), dim3(NEUR_BLOCK_SIZE)>>>(V_ms_dev, V_ms_last_dev, n_chs_dev, m_chs_dev, h_chs_dev, spike_times_dev, num_spikes_neur_dev,
				I_es_dev, ys_dev, I_syns_dev, y_psns_dev, I_psns_dev, psn_times_dev, psn_seeds_dev, exp_w_p_dev, exp_psc, rate, Nneur, t, h,
				Ds_dev, Inoise_dev, noise_states_dev, Vrec_dev);
		cudaDeviceSynchronize();
		integrate_synapses<<<dim3((Ncon + SYN_BLOCK_SIZE -1)/SYN_BLOCK_SIZE), dim3(SYN_BLOCK_SIZE)>>>(ys_dev, weights_dev, delays_dev, pre_conns_dev, post_conns_dev,
				spike_times_dev, num_spikes_syn_dev, num_spikes_neur_dev, t, Nneur, Ncon);
		cudaDeviceSynchronize();
    	if ((t % T_sim_partial) == 0){
			cerr << t*h << endl;
			CUDA_CHECK_RETURN(cudaMemcpy(spike_times, spike_times_dev, Nneur*sizeof(int)*T_sim_partial/time_part_syn, cudaMemcpyDeviceToHost));
			CUDA_CHECK_RETURN(cudaMemcpy(num_spikes_neur, num_spikes_neur_dev, Nneur*sizeof(int), cudaMemcpyDeviceToHost));
			CUDA_CHECK_RETURN(cudaMemcpy(num_spikes_syn, num_spikes_syn_dev, Ncon*sizeof(int), cudaMemcpyDeviceToHost));

			swap_spikes();
//			saveIVP2Fl();
			CUDA_CHECK_RETURN(cudaMemcpy(spike_times_dev, spike_times, Nneur*sizeof(int)*T_sim_partial/time_part_syn, cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(cudaMemcpy(num_spikes_neur_dev, num_spikes_neur, Nneur*sizeof(int), cudaMemcpyHostToDevice));
			CUDA_CHECK_RETURN(cudaMemcpy(num_spikes_syn_dev, num_spikes_syn, Ncon*sizeof(int), cudaMemcpyHostToDevice));
			if ( t % SaveIntervalTIdx == 0){
				apndResToFile();
				cerr << "Results saved to file!" << endl;
			}

		}
	}
#ifdef OSCILL_SAVE
	CUDA_CHECK_RETURN(cudaMemcpy(Vrec, Vrec_dev, Nneur*T_sim_partial/recInt*sizeof(float), cudaMemcpyDeviceToHost));
	save_oscill(0, true);
#endif

	cudaDeviceSynchronize();
	cudaMemcpy(spike_times, spike_times_dev, Nneur*sizeof(int)*T_sim_partial/time_part_syn, cudaMemcpyDeviceToHost);
	cudaMemcpy(num_spikes_neur, num_spikes_neur_dev, Nneur*sizeof(int), cudaMemcpyDeviceToHost);
	curr_time = time(0);
	cout << "Stop for rank: " << world_rank << " " << asctime(localtime(&curr_time)) << endl;

	save2HOST();
	apndResToFile();
#ifdef WITH_MPI
	MPI_Finalize();
#endif
	return 0;
}

void init_conns_from_file(){
	int Ncon_part;

	ifstream con_file;
	con_file.open(par_f_name);
	con_file >> Ncon_part;
	Ncon = Ncon_part*W_P_NUM_BUND*NUM_BUND;
	malloc_conn_memory();
	float delay;
	int pre, post;

	for (int s = 0; s < Ncon_part; s++){
		con_file >> pre >> post >> delay;
		for (unsigned int bund = 0; bund < W_P_NUM_BUND*NUM_BUND; bund++){
			int idx = bund*Ncon_part + s;
			pre_conns[idx] = pre + bund*BUND_SZ;
			post_conns[idx] = post + bund*BUND_SZ;
			delays[idx] = delay/h;
			weights[idx] = (expf(1.0f)/tau_psc)*w_n;
		}
	}
	con_file.close();
    if (strcmp(ivp_fname, "") != 0){
        cerr <<  "Loading connections from file" << endl;
        char fname[500];
        sprintf(fname, "%s/num_sp_syn", ivp_fname);
        file2array(fname, Ncon, num_spikes_syn);
    }
}

void init_neurs_from_file(){
	malloc_neur_memory();

	for (unsigned int bund = 0; bund < W_P_NUM_BUND; bund++){
		for (unsigned int n = 0; n < W_P_BUND_SZ; n++){
			unsigned int idx = W_P_BUND_SZ*bund + n;

			// IV on limit cycle
//			V_ms[idx] = 32.9066f;
//			V_ms_last[idx] = 32.9065f;
//			n_chs[idx] = 0.574678f;
//			m_chs[idx] = 0.913177f;
//			h_chs[idx] = 0.223994f;

			// IV at equilibrium state
			V_ms[idx] = -60.8457f;
			V_ms_last[idx] = -60.8450f;
			n_chs[idx] = 0.3763f;
			m_chs[idx] = 0.0833f;
			h_chs[idx] = 0.4636f;

			I_es[idx] = I_e;

			if (gaussNoiseFlag == 1){
				exp_w_p[idx] = 0.0f;
				Ds_host[idx] = (w_p_start + ((w_p_stop - w_p_start)/W_P_NUM_BUND)*bund);
//				Ds_host[idx] = pow(10, (w_p_start + ((w_p_stop - w_p_start)/W_P_NUM_BUND)*bund));
			} else {
				exp_w_p[idx] = (expf(1.0f)/tau_psc)*(w_p_start + ((w_p_stop - w_p_start)/W_P_NUM_BUND)*bund);
				Ds_host[idx] = 0.0f;
			}
		}
	}
	if (strcmp(ivp_fname, "") != 0){
	    char fname[500];
        cerr << "Loading neurons from files" << endl;

	    sprintf(fname, "%s/Vms", ivp_fname);
	    raw2array(fname, Nneur, V_ms);
	    sprintf(fname, "%s/Vms_last", ivp_fname);
	    raw2array(fname, Nneur, V_ms_last);

	    sprintf(fname, "%s/n_chs", ivp_fname);
	    raw2array(fname, Nneur, n_chs);
	    sprintf(fname, "%s/m_chs", ivp_fname);
	    raw2array(fname, Nneur, m_chs);
	    sprintf(fname, "%s/h_chs", ivp_fname);
	    raw2array(fname, Nneur, h_chs);

	    sprintf(fname, "%s/ys", ivp_fname);
	    raw2array(fname, Nneur, ys);
	    sprintf(fname, "%s/I_syns", ivp_fname);
	    raw2array(fname, Nneur, I_syns);

	    sprintf(fname, "%s/I_psns", ivp_fname);
	    raw2array(fname, Nneur, I_psns);
	    sprintf(fname, "%s/y_psns", ivp_fname);
	    raw2array(fname, Nneur, y_psns);

	    sprintf(fname, "%s/psn_times", ivp_fname);
	    file2array(fname, Nneur, psn_times);
	    sprintf(fname, "%s/psn_seeds", ivp_fname);
	    file2array(fname, Nneur, psn_seeds);

	    sprintf(fname, "%s/spike_times", ivp_fname);
	    FILE* ivpFl = fopen(fname, "r");

	    for (unsigned int n = 0; n < Nneur; n++){
	        fscanf(ivpFl, "%i", &num_spikes_neur[n]);
	        for (int sp_n = 0; sp_n < num_spikes_neur[n]; sp_n++){
	            fscanf(ivpFl, " %i", &spike_times[Nneur*sp_n + n]);
	        }
	    }
	    fclose(ivpFl);
	}
}

void save2HOST(){
	int w_p_bund_idx, w_p_bund_neur, bund_idx, idx, neur;
	for (unsigned int n = 0; n < Nneur; n++){
		w_p_bund_idx = n/W_P_BUND_SZ;
		w_p_bund_neur = n % W_P_BUND_SZ;
		bund_idx = w_p_bund_neur/BUND_SZ;
		neur = w_p_bund_neur % BUND_SZ;
		idx = NUM_BUND*w_p_bund_idx + bund_idx;
		for (int sp_n = 0; sp_n < num_spikes_neur[n]; sp_n++){
			res_senders[W_P_NUM_BUND*NUM_BUND*num_spk_in_bund[idx] + idx] = neur;
			res_times[W_P_NUM_BUND*NUM_BUND*num_spk_in_bund[idx] + idx] = spike_times[Nneur*sp_n + n]*h;
			num_spk_in_bund[idx]++;
		}
	}
}

void swap_spikes(){
	int* spike_times_temp = new int[Nneur*T_sim_partial/time_part_syn];
	unsigned int* min_spike_nums_syn = new unsigned int[Nneur];
	for (unsigned int n = 0; n < Nneur; n++){
		min_spike_nums_syn[n] = INT_MAX;
	}
	for (unsigned int s = 0; s < Ncon; s++){
		if (num_spikes_syn[s] < min_spike_nums_syn[pre_conns[s]]){
			min_spike_nums_syn[pre_conns[s]] = num_spikes_syn[s];
		}
	}
	// В случае если у нейрона не было никаких исходящих связей, то минимальное количество
	// Спйков которые обработли его исходящие синапсы будет равна INT_MAX, а это неверно
	// Поэтома надо насильно поставить 0, для этого тут и эта конструкция
	for (unsigned int n = 0; n < Nneur; n++){
		if (min_spike_nums_syn[n] == INT_MAX){
			min_spike_nums_syn[n] = 0;
		}
	}

	int w_p_bund_idx, w_p_bund_neur, bund_idx, neur, idx;
	for (unsigned int n = 0; n < Nneur; n++){
		w_p_bund_idx = n/W_P_BUND_SZ;
		w_p_bund_neur = n % W_P_BUND_SZ;
		bund_idx = w_p_bund_neur/BUND_SZ;
		neur = w_p_bund_neur % BUND_SZ;
		idx = NUM_BUND*w_p_bund_idx + bund_idx;
		for (unsigned int sp_n = 0; sp_n < min_spike_nums_syn[n]; sp_n++){
//		for (unsigned int sp_n = 0; sp_n < num_spikes_neur[n]; sp_n++){
			res_senders[W_P_NUM_BUND*NUM_BUND*num_spk_in_bund[idx] + idx] = neur;
			res_times[W_P_NUM_BUND*NUM_BUND*num_spk_in_bund[idx] + idx] = spike_times[Nneur*sp_n + n]*h;
			num_spk_in_bund[idx]++;
		}

		for (int sp_n = min_spike_nums_syn[n]; sp_n < num_spikes_neur[n]; sp_n++){
//		for (unsigned int sp_n = num_spikes_neur[n]; sp_n < num_spikes_neur[n]; sp_n++){
			spike_times_temp[Nneur*(sp_n - min_spike_nums_syn[n]) + n] = spike_times[Nneur*sp_n + n];
		}
		// @TODO В случае если считаем для несвязанных нейронов нужно убрать это
		 num_spikes_neur[n] -= min_spike_nums_syn[n];
//		 num_spikes_neur[n] = 0;
	}

	for (unsigned int s = 0; s < Ncon; s++){
		num_spikes_syn[s] -= min_spike_nums_syn[pre_conns[s]];
	}

	delete[] spike_times;
	delete[] min_spike_nums_syn;
	spike_times = spike_times_temp;
}

void saveIVP2Fl(){
    char ivpFl_name[500];
    sprintf(ivpFl_name, "%s/%i/spike_times", f_name, One);
    FILE* ivpFl = fopen(ivpFl_name, "w");

    for (unsigned int n = 0; n < Nneur; n++){
        fprintf(ivpFl, "%i", num_spikes_neur[n]);
        for (int sp_n = 0; sp_n < num_spikes_neur[n]; sp_n++){
            fprintf(ivpFl, " %i", spike_times[Nneur*sp_n + n]);
        }
        fprintf(ivpFl, "\n");
    }
    fclose(ivpFl);

    sprintf(ivpFl_name, "%s/%i/num_sp_syn", f_name, One);
    array2file(ivpFl_name, Ncon, num_spikes_syn);

    size_t n_fsize = Nneur*sizeof(float);

    cudaMemcpy(V_ms, V_ms_dev, n_fsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(V_ms_last, V_ms_last_dev, n_fsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(m_chs, m_chs_dev, n_fsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(n_chs, n_chs_dev, n_fsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_chs, h_chs_dev, n_fsize, cudaMemcpyDeviceToHost);

    cudaMemcpy(ys, ys_dev, n_fsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(I_syns, I_syns_dev, n_fsize, cudaMemcpyDeviceToHost);

    cudaMemcpy(I_psns, I_psns_dev, n_fsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(y_psns, y_psns_dev, n_fsize, cudaMemcpyDeviceToHost);

    cudaMemcpy(psn_times, psn_times_dev, Nneur*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(psn_seeds, psn_seeds_dev, Nneur*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    sprintf(ivpFl_name, "%s/%i/Vms", f_name, One);
    raw2file(ivpFl_name, Nneur, V_ms);
    sprintf(ivpFl_name, "%s/%i/Vms_last", f_name, One);
    raw2file(ivpFl_name, Nneur, V_ms_last);

    sprintf(ivpFl_name, "%s/%i/m_chs", f_name, One);
    raw2file(ivpFl_name, Nneur, m_chs);
    sprintf(ivpFl_name, "%s/%i/n_chs", f_name, One);
    raw2file(ivpFl_name, Nneur, n_chs);
    sprintf(ivpFl_name, "%s/%i/h_chs", f_name, One);
    raw2file(ivpFl_name, Nneur, h_chs);

    sprintf(ivpFl_name, "%s/%i/ys", f_name, One);
    raw2file(ivpFl_name, Nneur, ys);
    sprintf(ivpFl_name, "%s/%i/I_syns", f_name, One);
    raw2file(ivpFl_name, Nneur, I_syns);

    sprintf(ivpFl_name, "%s/%i/I_psns", f_name, One);
    raw2file(ivpFl_name, Nneur, I_psns);
    sprintf(ivpFl_name, "%s/%i/y_psns", f_name, One);
    raw2file(ivpFl_name, Nneur, y_psns);

    sprintf(ivpFl_name, "%s/%i/psn_times", f_name, One);
    array2file(ivpFl_name, Nneur, psn_times);

    sprintf(ivpFl_name, "%s/%i/psn_seeds", f_name, One);
    array2file(ivpFl_name, Nneur, psn_seeds);
    One += 1;
}

void array2file(char* fl_name, unsigned int N, unsigned int arr[]){
    FILE* fl = fopen(fl_name, "w");
    for (unsigned int i = 0; i < N; i++){
        fprintf(fl, "%u\n", arr[i]);
    }
    fclose(fl);
}

void file2array(char* fl_name, unsigned int N, unsigned int arr[]){
    FILE* fl = fopen(fl_name, "r");
    for (unsigned int i = 0; i < N; i++){
        fscanf(fl, "%u", &arr[i]);
    }
    fclose(fl);
}

void raw2file(char* fl_name, unsigned int N, float arr[]){
    FILE* fl = fopen(fl_name, "w");
    fwrite(arr, sizeof(float), N, fl);
    fclose(fl);
}

void raw2array(char* fl_name, unsigned int N, float arr[]){
    FILE* fl = fopen(fl_name, "r");
    fread(arr, sizeof(float), N, fl);
    fclose(fl);
}

void clearResFiles(){
	FILE* file;
	stringstream s;
	s.precision(3);
	char* name = new char[500];
	for (unsigned int i = 0; i < W_P_NUM_BUND; i++){
		for (unsigned int j = 0; j < NUM_BUND; j++){
			s << f_name << "/" << "seed_" << j + seed
					    << "/w_p_" << fixed << w_p_start + (w_p_stop - w_p_start)*i/W_P_NUM_BUND << endl;
			s >> name;
			file = fopen(name, "w");
			fclose(file);
		}
	}
 	delete[] name;
}

void apndResToFile(){
	FILE* file;
	stringstream s;
	s.precision(3);
	char* name = new char[500];
	for (unsigned int i = 0; i < W_P_NUM_BUND; i++){
		for (unsigned int j = 0; j < NUM_BUND; j++){
			s << f_name << "/" << "seed_" << j + seed
					    << "/w_p_" << fixed << w_p_start + (w_p_stop - w_p_start)*i/W_P_NUM_BUND << endl;
			s >> name;
			file = fopen(name, "a+");
			int idx = NUM_BUND*i + j;
			for (int spk = 0; spk < num_spk_in_bund[idx]; spk++){
				fprintf(file, "%i\t%.3f\n", res_senders[W_P_NUM_BUND*NUM_BUND*spk + idx], res_times[W_P_NUM_BUND*NUM_BUND*spk + idx]);
			}
			num_spk_in_bund[idx] = 0;
			fclose(file);
		}
	}
 	delete[] name;
}

void malloc_neur_memory(){
	V_ms = new float[Nneur];
	V_ms_last = new float[Nneur];
	m_chs = new float[Nneur];
	n_chs = new float[Nneur];
	h_chs = new float[Nneur];
	I_es = new float[Nneur];

	ys = new float[Nneur]();
	I_syns = new float[Nneur]();
	y_psns = new float[Nneur]();
	I_psns = new float[Nneur]();

	psn_times = new unsigned int[Nneur]();
	psn_seeds = new unsigned int[Nneur]();

	exp_w_p = new float[Nneur];

	Ds_host = new float[Nneur];

	// if num-th spike occur at a time t on n-th neuron then,
	// t is stored in element with index Nneur*num + n
	// spike_times[Nneur*num + n] = t
	spike_times = new int[Nneur*T_sim_partial/time_part_syn]();
	num_spikes_neur = new int[Nneur]();
	int expected_spk_num = BUND_SZ*SaveIntervalTIdx/time_part_syn;

	res_times = new float[W_P_NUM_BUND*NUM_BUND*expected_spk_num];
	res_senders = new int[W_P_NUM_BUND*NUM_BUND*expected_spk_num];
	num_spk_in_bund = new int[W_P_NUM_BUND*NUM_BUND]();
#ifdef OSCILL_SAVE
	Vrec = new float[Nneur*T_sim_partial/recInt];
#endif
}

void malloc_conn_memory(){
	weights = new float[Ncon];
	pre_conns = new int[Ncon];
	post_conns = new int[Ncon];
	delays = new int[Ncon];
	num_spikes_syn = new unsigned int[Ncon]();
}

void copy2device(){
	size_t n_fsize = Nneur*sizeof(float);
	size_t n_isize = Nneur*sizeof(int);
	size_t s_fsize = Ncon*sizeof(float);
	size_t s_isize = Ncon*sizeof(int);
	size_t spike_times_sz = n_isize*T_sim_partial/time_part_syn;

	// Allocating memory for array which contain var's for each neuron
	CUDA_CHECK_RETURN(cudaMalloc((void**) &V_ms_dev, n_fsize));
	cudaMalloc((void**) &V_ms_last_dev, n_fsize);
	cudaMalloc((void**) &m_chs_dev, n_fsize);
	cudaMalloc((void**) &n_chs_dev, n_fsize);
	cudaMalloc((void**) &h_chs_dev, n_fsize);
	cudaMalloc((void**) &I_es_dev, n_fsize);

	cudaMalloc((void**) &ys_dev, n_fsize);
	cudaMalloc((void**) &I_syns_dev, n_fsize);
	cudaMalloc((void**) &y_psns_dev, n_fsize);
	cudaMalloc((void**) &I_psns_dev, n_fsize);

	cudaMalloc((void**) &exp_w_p_dev, n_fsize);
	cudaMalloc((void**) &spike_times_dev, spike_times_sz);
	cudaMalloc((void**) &num_spikes_neur_dev, n_isize);

	cudaMalloc((void**) &psn_times_dev, n_isize);
	cudaMalloc((void**) &psn_seeds_dev, n_isize);

	// colored Gauss noise variables
	cudaMalloc((void**) &Ds_dev, n_fsize);
	cudaMalloc((void**) &Inoise_dev, n_fsize);
	cudaMalloc((void**) &noise_states_dev, Nneur*sizeof(curandState));

	cudaMemset(Inoise_dev, 0, n_fsize);
	cudaMemset(Ds_dev, 0, n_fsize);

	// Allocating memory for array which contain var's for each synapse
	cudaMalloc((void**) &weights_dev, s_fsize);
	cudaMalloc((void**) &pre_conns_dev, s_isize);
	cudaMalloc((void**) &post_conns_dev, s_isize);
	cudaMalloc((void**) &delays_dev, s_isize);
	cudaMalloc((void**) &num_spikes_syn_dev, s_isize);
#ifdef OSCILL_SAVE
	cudaMalloc((void**) &Vrec_dev, Nneur*T_sim_partial/recInt*sizeof(float));
#endif
	// Copying to GPU device memory neuron arrays
	cudaMemcpy(V_ms_dev, V_ms, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(V_ms_last_dev, V_ms_last, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(m_chs_dev, m_chs, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(n_chs_dev, n_chs, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(h_chs_dev, h_chs, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(I_es_dev, I_es, n_fsize, cudaMemcpyHostToDevice);

	cudaMemcpy(ys_dev, ys, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(I_syns_dev, I_syns, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(I_psns_dev, I_psns, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(y_psns_dev, y_psns, n_fsize, cudaMemcpyHostToDevice);

	cudaMemcpy(exp_w_p_dev, exp_w_p, n_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(Ds_dev, Ds_host, n_fsize, cudaMemcpyHostToDevice);

	cudaMemcpy(spike_times_dev, spike_times, spike_times_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(num_spikes_neur_dev, num_spikes_neur, n_isize, cudaMemcpyHostToDevice);

    cudaMemcpy(psn_times_dev, psn_times, n_isize, cudaMemcpyHostToDevice);
    cudaMemcpy(psn_seeds_dev, psn_seeds, Nneur*sizeof(unsigned int), cudaMemcpyHostToDevice);

	cudaMemcpy(weights_dev, weights, s_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(pre_conns_dev, pre_conns, s_isize, cudaMemcpyHostToDevice);
	cudaMemcpy(post_conns_dev, post_conns, s_isize, cudaMemcpyHostToDevice);
	cudaMemcpy(delays_dev, delays, s_isize, cudaMemcpyHostToDevice);
	cudaMemcpy(num_spikes_syn_dev, num_spikes_syn, s_isize, cudaMemcpyHostToDevice);
}

void init_params(int argc, char* argv[]){
	stringstream str;
	for (int i = 1; i < argc; i++){
		str << argv[i] << endl;
		switch (i){
			case 1: str >> SimulationTime; break;
			case 2: str >> h; break;
			case 3: str >> Nneur; break;
			case 4: str >> W_P_NUM_BUND; break;
			case 5: str >> BUND_SZ; break;
			case 6: str >> f_name; break;
			case 7: str >> seed; break;
			case 8: str >> rate; break;
			case 9: str >> w_p_start; break;
			case 10: str >> w_p_stop; break;
			case 11: str >> w_n; break;
			case 12: str >> par_f_name; break;
			case 13: str >> I_e; break;
			case 14: str >> dI_e; break;
			case 15: str >> gaussNoiseFlag; break;
			case 16: str >> ivp_fname; break;
			case 17: str >> Tstart; break;
		}
	}
	W_P_BUND_SZ = Nneur/W_P_NUM_BUND;
	NUM_BUND = W_P_BUND_SZ/BUND_SZ;
}

void clear_oscill_file(){
	FILE* file;
	stringstream s;
	s.precision(2);
	char* name = new char[500];
	for (unsigned int j = 0; j < Nneur; j++){
		s << f_name << "/" << "N_" << j << "_oscill" << endl;
		s >> name;
		file = fopen(name, "w");
		fclose(file);
	}
 	delete[] name;
}

void save_oscill(int tm, bool lastFlag /*lastFlag=false*/){
	int Tmax = T_sim_partial/recInt;
	if (lastFlag) {
		Tmax = ((T_sim - 1) % T_sim_partial)/recInt;
	}
	int Tstart_ = 0;
	if (tm == T_sim_partial){
	    Tstart_ = 1;
	}
	FILE* file;
	stringstream s;
	s.precision(2);
	char* name = new char[500];
	for (unsigned int j = 0; j < Nneur; j++){
		s << f_name << "/" << "N_" << j << "_oscill" << endl;
		s >> name;
		file = fopen(name, "a+");
		for (int t = Tstart_; t < Tmax; t++){
			fwrite(&Vrec[Nneur*t + j], sizeof(float), 1, file);
		}
		fclose(file);
	}
 	delete[] name;
}
