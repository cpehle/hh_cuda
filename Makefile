HEAD=$(shell git log -1 --format="%h")

all: hh-cuda-$(HEAD)

hh-cuda-$(HEAD): hodgkin-huxley-cuda-neuronet.cu hodgkin-huxley-cuda-neuronet.h
	nvcc -arch sm_50 -O3 --use_fast_math hodgkin-huxley-cuda-neuronet.cu -o hh-cuda-$(HEAD)
