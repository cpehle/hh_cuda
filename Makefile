HEAD=$(shell git log -1 --format="%h")

all: hh-cuda-$(HEAD)

hh-cuda-$(HEAD): hodgkin-huxley-cuda-neuronet.cu hodgkin-huxley-cuda-neuronet.h
	nvcc -arch sm_21 -O3 -Xcompiler -Wall --use_fast_math -DOSCILL_SAVE hodgkin-huxley-cuda-neuronet.cu -o hh-cuda-$(HEAD)
#	nvcc -arch sm_21 -Xcompiler -g3 -O0 -DOSCILL_SAVE hodgkin-huxley-cuda-neuronet.cu -o hh-cuda-$(HEAD)
#	nvcc -arch sm_21 --use_fast_math hodgkin-huxley-cuda-neuronet.cu -o hh-cuda-$(HEAD)

clean:
	rm hh-cuda-$(HEAD)