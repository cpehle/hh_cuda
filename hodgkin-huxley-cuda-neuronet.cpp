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
        // @TODO hardcode! 2-3 - number of GPU devices on each node of Lobachevsky Supercomputer 
        case 3: cudaSetDevice(world_rank % 3); break;
		case 2: cudaSetDevice(world_rank % 2); break;
        case 1: cudaSetDevice(0); break;
        default: exit(EXIT_FAILURE);
    }
    //I_e += dI_e*world_rank;
    seed += 150*world_rank;
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
        init_poisson(dim3(Nneur/NEUR_BLOCK_SIZE + 1), dim3(NEUR_BLOCK_SIZE), psn_times_dev, psn_seeds_dev, seed, rate, h, Nneur, W_P_BUND_SZ);
	}
	init_noise(dim3(Nneur/NEUR_BLOCK_SIZE + 1), dim3(NEUR_BLOCK_SIZE), noise_states_dev, Inoise_dev, Ds_dev, seed, Nneur, W_P_BUND_SZ);

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
		integrate_neurons(dim3((Nneur + NEUR_BLOCK_SIZE - 1)/NEUR_BLOCK_SIZE), dim3(NEUR_BLOCK_SIZE), V_ms_dev, V_ms_last_dev, n_chs_dev, m_chs_dev, h_chs_dev, spike_times_dev, num_spikes_neur_dev,
				I_es_dev, ys_dev, I_syns_dev, y_psns_dev, I_psns_dev, psn_times_dev, psn_seeds_dev, exp_w_p_dev, exp_psc, rate, Nneur, t, h,
				Ds_dev, Inoise_dev, noise_states_dev, Vrec_dev);
		cudaDeviceSynchronize();
		integrate_synapses(dim3((Ncon + SYN_BLOCK_SIZE -1)/SYN_BLOCK_SIZE), dim3(SYN_BLOCK_SIZE), ys_dev, weights_dev, delays_dev, pre_conns_dev, post_conns_dev,
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
    FILE* fl = fopen(fl_name, "wb");
    fwrite(arr, sizeof(float), N, fl);
    fclose(fl);
}

void raw2array(char* fl_name, unsigned int N, float arr[]){
    FILE* fl = fopen(fl_name, "rb");
    fread(arr, sizeof(float), N, fl);
    fclose(fl);
}

//void clearResFiles(){
//	FILE* file;
//	stringstream s;
//	s.precision(3);
//	char* name = new char[500];
//	for (unsigned int i = 0; i < W_P_NUM_BUND; i++){
//		for (unsigned int j = 0; j < NUM_BUND; j++){
//			s << f_name << "/" << "seed_" << j + seed
//					    << "/w_p_" << fixed << w_p_start + (w_p_stop - w_p_start)*i/W_P_NUM_BUND << endl;
//			s >> name;
//			file = fopen(name, "w");
//			fclose(file);
//		}
//	}
// 	delete[] name;
//}

void clearResFiles(){
	FILE* file;
	char* name = new char[500];
	for (unsigned int i = 0; i < W_P_NUM_BUND; i++){
		for (unsigned int j = 0; j < NUM_BUND; j++){
			float w_p = w_p_start + (w_p_stop - w_p_start)*i/W_P_NUM_BUND;

			sprintf(name, "%s/seed_%i/w_p_%.3f_senders", f_name, seed + j, w_p);
			file = fopen(name, "w");
			fclose(file);

			sprintf(name, "%s/seed_%i/w_p_%.3f_times", f_name, seed + j, w_p);
			file = fopen(name, "w");
			fclose(file);
		}
	}
 	delete[] name;
}

//void apndResToFile(){
//	FILE* file;
//	stringstream s;
//	s.precision(3);
//	char* name = new char[500];
//	for (unsigned int i = 0; i < W_P_NUM_BUND; i++){
//		for (unsigned int j = 0; j < NUM_BUND; j++){
//			s << f_name << "/" << "seed_" << j + seed
//					    << "/w_p_" << fixed << w_p_start + (w_p_stop - w_p_start)*i/W_P_NUM_BUND << endl;
//			s >> name;
//			file = fopen(name, "a+");
//			int idx = NUM_BUND*i + j;
//			for (int spk = 0; spk < num_spk_in_bund[idx]; spk++){
//				fprintf(file, "%i\t%.3f\n", res_senders[W_P_NUM_BUND*NUM_BUND*spk + idx], res_times[W_P_NUM_BUND*NUM_BUND*spk + idx]);
//			}
//			num_spk_in_bund[idx] = 0;
//			fclose(file);
//		}
//	}
// 	delete[] name;
//}

void apndResToFile(){
	FILE* file_times;
	FILE* file_senders;
	char* name_times = new char[500];
	char* name_senders = new char[500];
	for (unsigned int i = 0; i < W_P_NUM_BUND; i++){
		for (unsigned int j = 0; j < NUM_BUND; j++){
			float w_p = w_p_start + (w_p_stop - w_p_start)*i/W_P_NUM_BUND;

			sprintf(name_times, "%s/seed_%i/w_p_%.3f_times", f_name, seed + j, w_p);
			sprintf(name_senders, "%s/seed_%i/w_p_%.3f_senders", f_name, seed + j, w_p);

			file_times = fopen(name_times, "a+b");
			file_senders = fopen(name_senders, "a+b");
			int idx = NUM_BUND*i + j;
			for (int spk = 0; spk < num_spk_in_bund[idx]; spk++){
				fwrite(&res_times[W_P_NUM_BUND*NUM_BUND*spk + idx], sizeof(float), 1, file_times);
				fwrite(&res_senders[W_P_NUM_BUND*NUM_BUND*spk + idx], sizeof(int), 1, file_senders);
			}
			num_spk_in_bund[idx] = 0;
			fclose(file_times);
			fclose(file_senders);
		}
	}
 	delete[] name_times;
 	delete[] name_senders;
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
		file = fopen(name, "a+b");
		for (int t = Tstart_; t < Tmax; t++){
			fwrite(&Vrec[Nneur*t + j], sizeof(float), 1, file);
		}
		fclose(file);
	}
 	delete[] name;
}
