#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <climits>
#include "hodgkin-huxley-cuda-neuronet.h"

#define BLOCK_SIZE 64

float h = 0.05f;
float SimulationTime = 5000000.0f; // in ms
unsigned int seed = 1;

int T_sim_particular = 10000; // in time frames
// T_sim_particular must be less than RAND_MAX which in Win32 is 32767, in gcc much greater

int time_part_syn = 100;
// maximum part of simulating time for which is allocated memory
// time_part_syn <= T[ms]/h[ms]
// so the maximum number of spikes per neuron which can be processed is
// T_sim_particular/time_part_syn

int Nneur = 4000;
int NumBundle = 40;
int BundleSize = Nneur/NumBundle;

using namespace std;

// neuron parameters
__constant__ float Cm    = 1.0f; //  inverse of membrane capacity, 1/pF
__constant__ float g_Na  = 120.0f; // nS
__constant__ float g_K   = 36.0f;
__constant__ float g_L   = .3f;
__constant__ float E_K   = -77.0f;
__constant__ float E_Na  = 55.0f;
__constant__ float E_L   = -54.4f;
__constant__ float V_peak = 18.0f;

//connection parameters
float I_e = 5.27f;
float w_p_start = 1.96f; // pA
float w_p_stop = 2.04f;
float w_n = 1.3f;
float rate = 178.3f;
float tau_psc = 0.2f;
float exp_psc = expf(-h/tau_psc);
char f_name[] = "0/";

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
__global__ void integrate_synapses(float* y, float* I_psc, float* I_syn, float* weight, int* delay, int* pre_conn, int* post_conn,
		int* spike_time, int* num_spike_syn, int* num_spike_neur, int t, float h, float exp_psc, int Nneur, int Ncon){
	int s = blockDim.x*blockIdx.x + threadIdx.x;
	if (s < Ncon){
		I_psc[s]  = (y[s]*h + I_psc[s])*exp_psc;
		y[s] *= exp_psc;
		int neur = pre_conn[s];
		// if we processed less spikes than there is in presynaptic neuron
		// we need to check are there new spikes at this moment of time
		if (num_spike_syn[s] < num_spike_neur[neur]){
			if (spike_time[Nneur*num_spike_syn[s] + neur] == t - delay[s]){
				y[s] += weight[s];
				num_spike_syn[s]++;
			}
		}
		atomicAdd(&I_syn[post_conn[s]], I_psc[s]);
	}
}

__global__ void integrate_neurons(
		float* V_m, float* V_m_last, float* n_ch, float* m_ch, float* h_ch,
		int* spike_time, int* num_spike_neur,
		float* I_e, float* y_psn, float* I_psn, int* psn_time, unsigned int* psn_seed,
		float* I_syn, float* I_syn_last, float* exp_w_p, float exp_psc, float rate,
		int Nneur, int t, float h){
		int n = blockIdx.x*blockDim.x + threadIdx.x;
		if (n < Nneur){
			I_psn[n]  = (y_psn[n]*h + I_psn[n])*exp_psc;
			y_psn[n] *= exp_psc;

			// if where is poisson impulse on neuron
			while (psn_time[n] == t){
				y_psn[n] += exp_w_p[n];
				psn_time[n] -= (1000.0f/(rate*h))*logf(get_random(psn_seed + n));
			}

			I_syn[n] += I_psn[n];

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

			v2 = hh_Vm(V_m[n], n_ch[n], m_ch[n], h_ch[n], (I_syn[n] + I_syn_last[n])/2.0f, I_e[n], h);
			n2 = hh_n_ch(V_m[n], n_ch[n], h);
			m2 = hh_m_ch(V_m[n], m_ch[n], h);
			h2 = hh_h_ch(V_m[n], h_ch[n], h);
			V_m[n] = V_mem + v2/2.0f;
			n_ch[n] = n_channel + n2/2.0f;
			m_ch[n] = m_channel + m2/2.0f;
			h_ch[n] = h_channel + h2/2.0f;

			v3 = hh_Vm(V_m[n], n_ch[n], m_ch[n], h_ch[n], (I_syn[n] + I_syn_last[n])/2.0f, I_e[n], h);
			n3 = hh_n_ch(V_m[n], n_ch[n], h);
			m3 = hh_m_ch(V_m[n], m_ch[n], h);
			h3 = hh_h_ch(V_m[n], h_ch[n], h);
			V_m[n] = V_mem + v3;
			n_ch[n] = n_channel + n3;
			m_ch[n] = m_channel + m3;
			h_ch[n] = h_channel + h3;

			v4 = hh_Vm(V_m[n], n_ch[n], m_ch[n], h_ch[n], I_syn[n], I_e[n], h);
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
			I_syn_last[n] = I_syn[n];
			I_syn[n] = 0.0f;
//			if (n == 0){
//				printf("%.2f; %g; %g; %g; %g; %g; %g; %g; %g; %g\n",
//						t*h, V_m[n], V_m[n+1], n_ch[n], m_ch[n], h_ch[n], I_psn[n], y_psn[n], I_syn_last[n], I_syn_last[n+1]);
//			}
		}
}

int main(int argc, char* argv[]){
	init_params(argc, argv);
	T_sim = SimulationTime/h;
	init_neurs_from_file();
	init_conns_from_file();
	copy2device();
	clear_files();

	init_poisson<<<dim3(Nneur/BLOCK_SIZE + 1), dim3(BLOCK_SIZE)>>>(psn_times_dev, psn_seeds_dev, seed, rate, h, Nneur, BundleSize);
	clock_t start = clock();
	for (int t = 1; t < T_sim; t++){
		integrate_neurons<<<dim3(Nneur/BLOCK_SIZE + 1), dim3(BLOCK_SIZE)>>>(V_ms_dev, V_ms_last_dev, n_chs_dev, m_chs_dev, h_chs_dev, spike_times_dev, num_spikes_neur_dev,
				I_es_dev, y_psns_dev, I_psns_dev, psn_times_dev, psn_seeds_dev, I_syns_dev, I_syns_last_dev, exp_w_p_dev, exp_psc, rate, Nneur, t, h);
		cudaDeviceSynchronize();
		integrate_synapses<<<dim3(Ncon/BLOCK_SIZE + 1), dim3(BLOCK_SIZE)>>>(ys_dev, I_syn_partial_dev, I_syns_dev, weights_dev, delays_dev, pre_conns_dev, post_conns_dev,
				spike_times_dev, num_spikes_syn_dev, num_spikes_neur_dev, t, h, exp_psc, Nneur, Ncon);
		cudaDeviceSynchronize();
		if ((t % T_sim_particular) == 0){
			cerr << t*h << endl;
			cudaMemcpy(spike_times, spike_times_dev, Nneur*sizeof(int)*T_sim_particular/time_part_syn, cudaMemcpyDeviceToHost);
			cudaMemcpy(num_spikes_neur, num_spikes_neur_dev, Nneur*sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(num_spikes_syn, num_spikes_syn_dev, Ncon*sizeof(int), cudaMemcpyDeviceToHost);
			swap_spikes();
			cudaMemcpy(spike_times_dev, spike_times, Nneur*sizeof(int)*T_sim_particular/time_part_syn, cudaMemcpyHostToDevice);
			cudaMemcpy(num_spikes_neur_dev, num_spikes_neur, Nneur*sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(num_spikes_syn_dev, num_spikes_syn, Ncon*sizeof(int), cudaMemcpyHostToDevice);
		}
	}
	cudaDeviceSynchronize();
	cudaMemcpy(spike_times, spike_times_dev, Nneur*sizeof(int)*T_sim_particular/time_part_syn, cudaMemcpyDeviceToHost);
	cudaMemcpy(num_spikes_neur, num_spikes_neur_dev, Nneur*sizeof(int), cudaMemcpyDeviceToHost);
	float time = ((float)clock() - (float)start)*1000./CLOCKS_PER_SEC;
	cerr << "Elapsed time: " << time << " ms" << endl;

	save2file();
	cerr << "Finished!" << endl;
	return 0;
}

void init_conns_from_file(){
	int Ncon_part;
	int Nneur_part = Nneur/NumBundle;

	ifstream con_file;
	con_file.open("nn_params.csv");
	con_file >> Ncon_part;
	Ncon = Ncon_part*NumBundle;
	cerr << "Number of connections: " << Ncon << endl;
	malloc_conn_memory();
	float delay;
	int pre, post;

	for (int s = 0; s < Ncon_part; s++){
		con_file >> pre >> post >> delay;
		for (int bund = 0; bund < NumBundle; bund++){
			int idx = bund*Ncon_part + s;
			pre_conns[idx] = pre + bund*Nneur_part;
			post_conns[idx] = post + bund*Nneur_part;
			delays[idx] = delay/h;
			weights[idx] = (expf(1.0f)/tau_psc)*w_n;
		}
	}
	con_file.close();
}

void init_neurs_from_file(){
	srand(0);
	malloc_neur_memory();
	for (int bund = 0; bund < NumBundle; bund++){
		for (int n = 0; n < BundleSize; n++){
			int idx = BundleSize*bund + n;
			V_ms[idx] = 32.9066f;
			V_ms_last[idx] = 32.9065f;
			n_chs[idx] = 0.574678f;
			m_chs[idx] = 0.913177f;
			h_chs[idx] = 0.223994f;
			I_es[idx] = I_e;
			exp_w_p[idx] = (expf(1.0f)/tau_psc)*(w_p_start + (w_p_stop - w_p_start)*bund/NumBundle);
		}
	}
}

void save2file(){
	FILE** files = new FILE*[NumBundle];
	stringstream s;
	char* name = new char[500];
	for (int i = 0; i < NumBundle; i++){
		s << f_name << "w_p_" << w_p_start + (w_p_stop - w_p_start)*i/NumBundle << endl;
		s >> name;
		files[i] = fopen(name, "a");
	}
	int idx, neur;
	for (int n = 0; n < Nneur; n++){
		idx = n/BundleSize;
		neur = n - BundleSize*idx;
		for (int sp_n = 0; sp_n < num_spikes_neur[n]; sp_n++){
			fprintf(files[idx], "%.1f; %i; \n", spike_times[Nneur*sp_n + n]*h, neur);
		}
	}

}

void swap_spikes(){
	FILE** files = new FILE*[NumBundle];
	stringstream s;
	char* name = new char[500];
	for (int i = 0; i < NumBundle; i++){
		s << f_name << "w_p_" << w_p_start + (w_p_stop - w_p_start)*i/NumBundle << endl;
		s >> name;
		files[i] = fopen(name, "a");
	}

	int* spike_times_temp = new int[Nneur*T_sim_particular/time_part_syn];
	int* min_spike_nums_syn = new int[Nneur];
	for (int n = 0; n < Nneur; n++){
		min_spike_nums_syn[n] = INT_MAX;
	}
	for (int s = 0; s < Ncon; s++){
		if (num_spikes_syn[s] < min_spike_nums_syn[pre_conns[s]]){
			min_spike_nums_syn[pre_conns[s]] = num_spikes_syn[s];
		}
	}
	int idx, neur;
	for (int n = 0; n < Nneur; n++){
		idx = n/BundleSize;
		neur = n - BundleSize*idx;
		for (int sp_n = 0; sp_n < min_spike_nums_syn[n]; sp_n++){
			fprintf(files[idx], "%.1f; %i; \n", spike_times[Nneur*sp_n + n]*h, neur);
		}

		for (int sp_n = min_spike_nums_syn[n]; sp_n < num_spikes_neur[n]; sp_n++){
			spike_times_temp[Nneur*(sp_n - min_spike_nums_syn[n]) + n] = spike_times[Nneur*sp_n + n];
		}
		num_spikes_neur[n] = num_spikes_neur[n] - min_spike_nums_syn[n];
	}

	for (int i = 0; i < NumBundle; i++){
		fclose(files[i]);
	}

	for (int s = 0; s < Ncon; s++){
		num_spikes_syn[s] = num_spikes_syn[s] - min_spike_nums_syn[pre_conns[s]];
	}

	free(spike_times);
	free(min_spike_nums_syn);
	spike_times = spike_times_temp;
}

void malloc_neur_memory(){
	V_ms = new float[Nneur];
	V_ms_last = new float[Nneur];
	m_chs = new float[Nneur];
	n_chs = new float[Nneur];
	h_chs = new float[Nneur];
	I_syns = new float[Nneur]();
	I_syns_last = new float[Nneur]();
	I_es = new float[Nneur];
	I_psns = new float[Nneur]();
	y_psns = new float[Nneur]();
	exp_w_p = new float[Nneur];
	// if num-th spike occur at a time t on n-th neuron then,
	// t is stored in element with index Nneur*num + n
	// spike_times[Nneur*num + n] = t
	spike_times = new int[Nneur*T_sim_particular/time_part_syn]();
	num_spikes_neur = new int[Nneur]();
}

void malloc_conn_memory(){
	ys = new float[Ncon];
	I_syn_partial = new float[Ncon];
	weights = new float[Ncon];
	pre_conns = new int[Ncon];
	post_conns = new int[Ncon];
	delays = new int[Ncon];
	num_spikes_syn = new int[Ncon];
	memset(ys, 0, Ncon*sizeof(int));
	memset(I_syn_partial, 0, Ncon*sizeof(int));
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
	cudaMalloc((void**) &exp_w_p_dev, n_fsize);

	cudaMalloc((void**) &psn_times_dev, n_isize);
	cudaMalloc((void**) &psn_seeds_dev, n_isize);

	cudaMalloc((void**) &spike_times_dev, n_isize*T_sim_particular/time_part_syn);
	cudaMalloc((void**) &num_spikes_neur_dev, n_isize);

	int s_fsize = Ncon*sizeof(float);
	int s_isize = Ncon*sizeof(int);
	cudaMalloc((void**) &ys_dev, s_fsize);
	cudaMalloc((void**) &I_syn_partial_dev, s_fsize);
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
	cudaMemcpy(exp_w_p_dev, exp_w_p, n_fsize, cudaMemcpyHostToDevice);

	cudaMemcpy(spike_times_dev, spike_times, n_isize*T_sim_particular/time_part_syn, cudaMemcpyHostToDevice);
	cudaMemcpy(num_spikes_neur_dev, num_spikes_neur, n_isize, cudaMemcpyHostToDevice);

	cudaMemcpy(ys_dev, ys, s_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(I_syn_partial_dev, I_syn_partial, s_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(weights_dev, weights, s_fsize, cudaMemcpyHostToDevice);
	cudaMemcpy(pre_conns_dev, pre_conns, s_isize, cudaMemcpyHostToDevice);
	cudaMemcpy(post_conns_dev, post_conns, s_isize, cudaMemcpyHostToDevice);
	cudaMemcpy(delays_dev, delays, s_isize, cudaMemcpyHostToDevice);
	cudaMemcpy(num_spikes_syn_dev, num_spikes_syn, s_isize, cudaMemcpyHostToDevice);
}

void clear_files(){
	FILE** files = new FILE*[NumBundle];
	stringstream s;
	char* name = new char[500];
	for (int i = 0; i < NumBundle; i++){
		s << f_name << "w_p_" << w_p_start + (w_p_stop - w_p_start)*i/NumBundle << endl;
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
			case 3: str >> w_p_start; break;
			case 4: str >> w_p_stop; break;
			case 5: str >> f_name; break;
		}
	}
}
