all: 
	nvcc -arch sm_21 -O3 hodgkin-huxley-cuda-neuronet.cu -o hodgkin-huxley-cuda-neuronet

