all: hodgkin-huxley-cuda-neuronet

hodgkin-huxley-cuda-neuronet: hodgkin-huxley-cuda-neuronet.cu hodgkin-huxley-cuda-neuronet.h
	/usr/local/cuda/bin/nvcc -arch sm_21 -O3 hodgkin-huxley-cuda-neuronet.cu -o hodgkin-huxley-cuda-neuronet
