#include <cstdio>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <climits>
#include <ctime>
#include "hodgkin-huxley-cuda-neuronet.h"

unsigned int seed = 1;
float h = 0.1f;
float SimulationTime = 10000.0f; // in ms

int Nneur = 20;
int W_P_NUM_BUND = 1; // number of different poisson weights
int W_P_BUND_SZ = Nneur/W_P_NUM_BUND; // Number of neurons in bundle with same w_ps
int BUND_SZ = 2;  // Number of neurons in a single realization
int NUM_BUND = W_P_BUND_SZ/BUND_SZ;

// connection parameters
float I_e = 5.27f;
float w_p_start = 1.8f; // pA
float w_p_stop = 2.0f;
float w_n = 5.4f;
float rate = 200.0f;

char f_name[500] = "0";
char par_f_name[500] = "0";

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
//      For understanding why what, calculate the limit for v/(1 - exp(v/10)) then v tend to 0
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

__global__ void init_poisson(int* psn_time, unsigned int *psn_seed, unsigned int seed, float rate, float h, int Nneur, int BundleSize){
	int n = blockIdx.x*blockDim.x + threadIdx.x;
	int neur = n % BundleSize;
	if (n < Nneur){
		psn_seed[n] = seed + 100000*(neur + 1);
		psn_time[n] = -(1000.0f/(h*rate))*logf(get_random(psn_seed + n));
	}
}

__global__ void integrate_synapses(float* y, float* weight, int* delay, int* pre_conn, int* post_conn,
		int* spike_time, int* num_spike_syn, int* num_spike_neur, int t, int Nneur, int Ncon){
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
		float* I_e, float* y, float* I_syn, float* y_psn, float* I_psn, int* psn_time, unsigned int* psn_seed,
		float* I_syn_last, float* exp_w_p, float exp_psc, float rate,
		int Nneur, int t, float h){
		int n = blockIdx.x*blockDim.x + threadIdx.x;
		if (n < Nneur){
			I_psn[n]  = (y_psn[n]*h + I_psn[n])*exp_psc;
			y_psn[n] *= exp_psc;

			I_syn[n]  = (y[n]*h + I_syn[n])*exp_psc;
			y[n] *= exp_psc;

			// if where is poisson impulse on neuron
			while (psn_time[n] == t){
				y_psn[n] += exp_w_p[n];
				psn_time[n] -= (1000.0f/(rate*h))*logf(get_random(psn_seed + n));
			}

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

			v2 = hh_Vm(V_m[n], n_ch[n], m_ch[n], h_ch[n], (I_syn[n] + I_psn[n] + I_syn_last[n])/2.0f, I_e[n], h);
			n2 = hh_n_ch(V_m[n], n_ch[n], h);
			m2 = hh_m_ch(V_m[n], m_ch[n], h);
			h2 = hh_h_ch(V_m[n], h_ch[n], h);
			V_m[n] = V_mem + v2/2.0f;
			n_ch[n] = n_channel + n2/2.0f;
			m_ch[n] = m_channel + m2/2.0f;
			h_ch[n] = h_channel + h2/2.0f;

			v3 = hh_Vm(V_m[n], n_ch[n], m_ch[n], h_ch[n], (I_syn[n] + I_psn[n] + I_syn_last[n])/2.0f, I_e[n], h);
			n3 = hh_n_ch(V_m[n], n_ch[n], h);
			m3 = hh_m_ch(V_m[n], m_ch[n], h);
			h3 = hh_h_ch(V_m[n], h_ch[n], h);
			V_m[n] = V_mem + v3;
			n_ch[n] = n_channel + n3;
			m_ch[n] = m_channel + m3;
			h_ch[n] = h_channel + h3;

			v4 = hh_Vm(V_m[n], n_ch[n], m_ch[n], h_ch[n], I_syn[n] + I_psn[n], I_e[n], h);
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
			I_syn_last[n] = I_syn[n] + I_psn[n];

//			if (n == 0){
//				printf("%.3f;%g;%g;%g;%g;%g;%g\n",
//						t*h, V_m[n], V_m[n+1], I_psn[n], I_psn[n+1], I_syn[n], I_syn[n+1]);
//			}
		}
}

using namespace std;

int main(int argc, char* argv[]){
	init_params(argc, argv);
	exp_psc = expf(-h/tau_psc);
	time_part_syn = 10.0f/h;
	T_sim = SimulationTime/h;
	init_neurs_from_file();
	init_conns_from_file();
	copy2device();
//	clear_files();
//	cudaError_t er;

	init_poisson<<<dim3(Nneur/NEUR_BLOCK_SIZE + 1), dim3(NEUR_BLOCK_SIZE)>>>(psn_times_dev, psn_seeds_dev, seed, rate, h, Nneur, W_P_BUND_SZ);
	clock_t start = clock();
	time_t curr_time = time(0);
    char* st = asctime(localtime(&curr_time));
	cerr << "Start: " << st << endl;
    for (int t = 1; t < T_sim; t++){
		integrate_neurons<<<dim3(Nneur/NEUR_BLOCK_SIZE + 1), dim3(NEUR_BLOCK_SIZE)>>>(V_ms_dev, V_ms_last_dev, n_chs_dev, m_chs_dev, h_chs_dev, spike_times_dev, num_spikes_neur_dev,
				I_es_dev, ys_dev, I_syns_dev, y_psns_dev, I_psns_dev, psn_times_dev, psn_seeds_dev, I_last_dev, exp_w_p_dev, exp_psc, rate, Nneur, t, h);
		cudaDeviceSynchronize();
		integrate_synapses<<<dim3(Ncon/SYN_BLOCK_SIZE + 1), dim3(SYN_BLOCK_SIZE)>>>(ys_dev, weights_dev, delays_dev, pre_conns_dev, post_conns_dev,
				spike_times_dev, num_spikes_syn_dev, num_spikes_neur_dev, t, Nneur, Ncon);
		cudaDeviceSynchronize();
//		if(er != cudaSuccess){
//			cerr << cudaGetErrorString(er) << endl;
//		}

		if ((t % T_sim_partial) == 0){
			cout << t*h << endl;
			CUDA_CHECK_RETURN(cudaMemcpy(spike_times, spike_times_dev, Nneur*sizeof(int)*T_sim_partial/time_part_syn, cudaMemcpyDeviceToHost));
			CUDA_CHECK_RETURN(cudaMemcpy(num_spikes_neur, num_spikes_neur_dev, Nneur*sizeof(int), cudaMemcpyDeviceToHost));
			CUDA_CHECK_RETURN(cudaMemcpy(num_spikes_syn, num_spikes_syn_dev, Ncon*sizeof(int), cudaMemcpyDeviceToHost));
			swap_spikes();
			cudaMemcpy(spike_times_dev, spike_times, Nneur*sizeof(int)*T_sim_partial/time_part_syn, cudaMemcpyHostToDevice);
			cudaMemcpy(num_spikes_neur_dev, num_spikes_neur, Nneur*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(num_spikes_syn_dev, num_spikes_syn, Ncon*sizeof(int), cudaMemcpyHostToDevice);
		}
	}
	cudaDeviceSynchronize();
	cudaMemcpy(spike_times, spike_times_dev, Nneur*sizeof(int)*T_sim_partial/time_part_syn, cudaMemcpyDeviceToHost);
	cudaMemcpy(num_spikes_neur, num_spikes_neur_dev, Nneur*sizeof(int), cudaMemcpyDeviceToHost);
	curr_time = time(0);
	cerr << "Stop: " << asctime(localtime(&curr_time)) << endl;
	cerr << "Finished!" << endl;

//	float s_time = ((float) clock() - (float) start)*1000./CLOCKS_PER_SEC;
//	cerr << "Elapsed time: " << s_time << " ms" << endl;
	save2HOST();
	save2file();
	return 0;
}

void init_conns_from_file(){
	int Ncon_part;

	ifstream con_file;
	con_file.open(par_f_name);
	con_file >> Ncon_part;
	Ncon = Ncon_part*W_P_NUM_BUND*NUM_BUND;
//	cerr << "Number of connections: " << Ncon << endl;
	malloc_conn_memory();
	float delay;
	int pre, post;

	for (int s = 0; s < Ncon_part; s++){
		con_file >> pre >> post >> delay;
		for (int bund = 0; bund < W_P_NUM_BUND*NUM_BUND; bund++){
			int idx = bund*Ncon_part + s;
			pre_conns[idx] = pre + bund*BUND_SZ;
			post_conns[idx] = post + bund*BUND_SZ;
			delays[idx] = delay/h;
			weights[idx] = (expf(1.0f)/tau_psc)*w_n;
		}
	}
	con_file.close();
}

void init_neurs_from_file(){
	unsigned int* param_distr_seeds = new unsigned int[Nneur]();
	for (int i = 0; i < Nneur; i++){
		int neur = i % BUND_SZ;
		param_distr_seeds[i] = 100000*(neur + 1);
		get_random(param_distr_seeds + i);
//		get_random(&param_distr_seeds[i]);
	}
	malloc_neur_memory();
	for (int bund = 0; bund < W_P_NUM_BUND; bund++){
		for (int n = 0; n < W_P_BUND_SZ; n++){
			int idx = W_P_BUND_SZ*bund + n;

			// IV on limit cycle
			V_ms[idx] = 32.9066f;
			V_ms_last[idx] = 32.9065f;
			n_chs[idx] = 0.574678f;
			m_chs[idx] = 0.913177f;
			h_chs[idx] = 0.223994f;

			// IV at equilibrium state
//			V_ms[idx] = -60.8457f;
//			V_ms_last[idx] = -60.8450f;
//			n_chs[idx] = 0.3763f;
//			m_chs[idx] = 0.0833f;
//			h_chs[idx] = 0.4636f;

//			unsigned int ivp_seed = seed + 1000 * n;
//			V_ms[idx] = -75.4989f + (32.9031f + 75.4989f) * get_random(&ivp_seed);
//			V_ms_last[idx] = V_ms[idx] - 0.001f;
//			n_chs[idx] = 0.3593f + (0.7574f - 0.3593f) * get_random(&ivp_seed);
//			m_chs[idx] = 0.0149f + (0.0149f - 0.9895f) * get_random(&ivp_seed);
//			h_chs[idx] = 0.0669f + (0.0669f - 0.5121f) * get_random(&ivp_seed);

//			I_es[idx] = I_e;
			float I_e_min = 5.22f;
			float I_e_max = 5.30f;
			I_es[idx] = I_e_min + (I_e_max-I_e_min)*get_random(param_distr_seeds + idx);
			exp_w_p[idx] = (expf(1.0f)/tau_psc)*(w_p_start + (w_p_stop - w_p_start)*bund/W_P_NUM_BUND);
		}
	}
}

void save2HOST(){
	int w_p_bund_idx, w_p_bund_neur, bund_idx, idx, neur;
	for (int n = 0; n < Nneur; n++){
		w_p_bund_idx = n/W_P_BUND_SZ;
		w_p_bund_neur = n - W_P_BUND_SZ*w_p_bund_idx;
		bund_idx = w_p_bund_neur/BUND_SZ;
		neur = w_p_bund_neur - BUND_SZ*bund_idx;
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
	int* min_spike_nums_syn = new int[Nneur];
	for (int n = 0; n < Nneur; n++){
		min_spike_nums_syn[n] = INT_MAX;
	}
	for (int s = 0; s < Ncon; s++){
		if (num_spikes_syn[s] < min_spike_nums_syn[pre_conns[s]]){
			min_spike_nums_syn[pre_conns[s]] = num_spikes_syn[s];
		}
	}
	// В случае если у нейрона не было никаких исходящих связей, то минимальное количество
	// Спйков которые обрботли его исходящие синапсы будет равна INT_MAX, а это неверно
	// Поэтома надо насильно поставить 0, для этого тут и эта конструкция
	for (int n = 0; n < Nneur; n++){
		if (min_spike_nums_syn[n] == INT_MAX){
			min_spike_nums_syn[n] = 0;
		}
	}

	int w_p_bund_idx, w_p_bund_neur, bund_idx, neur, idx;
	for (int n = 0; n < Nneur; n++){
		w_p_bund_idx = n/W_P_BUND_SZ;
		w_p_bund_neur = n - W_P_BUND_SZ*w_p_bund_idx;
		bund_idx = w_p_bund_neur/BUND_SZ;
		neur = w_p_bund_neur - BUND_SZ*bund_idx;
		idx = NUM_BUND*w_p_bund_idx + bund_idx;
		for (int sp_n = 0; sp_n < min_spike_nums_syn[n]; sp_n++){
			res_senders[W_P_NUM_BUND*NUM_BUND*num_spk_in_bund[idx] + idx] = neur;
			res_times[W_P_NUM_BUND*NUM_BUND*num_spk_in_bund[idx] + idx] = spike_times[Nneur*sp_n + n]*h;
			num_spk_in_bund[idx]++;
		}

		for (int sp_n = min_spike_nums_syn[n]; sp_n < num_spikes_neur[n]; sp_n++){
			spike_times_temp[Nneur*(sp_n - min_spike_nums_syn[n]) + n] = spike_times[Nneur*sp_n + n];
		}
		num_spikes_neur[n] = num_spikes_neur[n] - min_spike_nums_syn[n];
	}

	for (int s = 0; s < Ncon; s++){
		num_spikes_syn[s] = num_spikes_syn[s] - min_spike_nums_syn[pre_conns[s]];
	}

	free(spike_times);
	free(min_spike_nums_syn);
	spike_times = spike_times_temp;
}

void save2file(){
	FILE* file;
	stringstream s;
	char* name = new char[500];
	for (int i = 0; i < W_P_NUM_BUND; i++){
		for (int j = 0; j < NUM_BUND; j++){
			s << f_name << "/"<< j << "/w_p_" << w_p_start + (w_p_stop - w_p_start)*i/W_P_NUM_BUND << endl;
			s >> name;
			file = fopen(name, "w");
			int idx = NUM_BUND*i + j;
			for (int spk = 0; spk < num_spk_in_bund[idx]; spk++){
				fprintf(file, "%.3f;%i\n", res_times[W_P_NUM_BUND*NUM_BUND*spk + idx], res_senders[W_P_NUM_BUND*NUM_BUND*spk + idx]);
			}
			fclose(file);
		}
	}

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

	I_last = new float[Nneur]();

	exp_w_p = new float[Nneur];

	// if num-th spike occur at a time t on n-th neuron then,
	// t is stored in element with index Nneur*num + n
	// spike_times[Nneur*num + n] = t
	spike_times = new int[Nneur*T_sim_partial/time_part_syn]();
	num_spikes_neur = new int[Nneur]();
	int expected_spk_num = BUND_SZ*SimulationTime/5.0f;
	res_times = new float[W_P_NUM_BUND*NUM_BUND*expected_spk_num];
	res_senders = new int[W_P_NUM_BUND*NUM_BUND*expected_spk_num];
	num_spk_in_bund = new int[W_P_NUM_BUND*NUM_BUND]();
}

void malloc_conn_memory(){
	weights = new float[Ncon];
	pre_conns = new int[Ncon];
	post_conns = new int[Ncon];
	delays = new int[Ncon];
	num_spikes_syn = new int[Ncon]();
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

	cudaMalloc((void**) &I_last_dev, n_fsize);

	cudaMalloc((void**) &exp_w_p_dev, n_fsize);
	cudaMalloc((void**) &spike_times_dev, spike_times_sz);
	cudaMalloc((void**) &num_spikes_neur_dev, n_isize);

	cudaMalloc((void**) &psn_times_dev, n_isize);
	cudaMalloc((void**) &psn_seeds_dev, n_isize);

	// Allocating memory for array which contain var's for each synapse
	cudaMalloc((void**) &weights_dev, s_fsize);
	cudaMalloc((void**) &pre_conns_dev, s_isize);
	cudaMalloc((void**) &post_conns_dev, s_isize);
	cudaMalloc((void**) &delays_dev, s_isize);
	cudaMalloc((void**) &num_spikes_syn_dev, s_isize);

	// Copyng to GPU device memory neuron arrays
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

	cudaMemcpy(I_last_dev, I_last, n_fsize, cudaMemcpyHostToDevice);

	cudaMemcpy(exp_w_p_dev, exp_w_p, n_fsize, cudaMemcpyHostToDevice);

	cudaMemcpy(spike_times_dev, spike_times, spike_times_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(num_spikes_neur_dev, num_spikes_neur, n_isize, cudaMemcpyHostToDevice);

	cudaMemcpy(weights_dev, weights, s_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(pre_conns_dev, pre_conns, s_isize, cudaMemcpyHostToDevice);
	cudaMemcpy(post_conns_dev, post_conns, s_isize, cudaMemcpyHostToDevice);
	cudaMemcpy(delays_dev, delays, s_isize, cudaMemcpyHostToDevice);
	cudaMemcpy(num_spikes_syn_dev, num_spikes_syn, s_isize, cudaMemcpyHostToDevice);
}

void clear_files(){
	FILE** files = new FILE*[W_P_NUM_BUND];
	stringstream s;
	char* name = new char[500];
	for (int i = 0; i < W_P_NUM_BUND; i++){
		s << f_name << "/w_p_" << w_p_start + (w_p_stop - w_p_start)*i/W_P_NUM_BUND << endl;
		s >> name;
		files[i] = fopen(name, "w");
		fclose(files[i]);
	}
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
		}
	}
	W_P_BUND_SZ = Nneur/W_P_NUM_BUND;
	NUM_BUND = W_P_BUND_SZ/BUND_SZ;
}
