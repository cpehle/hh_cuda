HEAD=$(shell git log -1 --format="%h")
MPI_INC=-I/usr/lib/openmpi/include/openmpi -I/usr/lib/openmpi/include
MPI_LIB_PATH=-L/usr/lib -L/usr/lib/openmpi/lib
MPI_LIB=-lmpi_cxx -lmpi -ldl -lhwloc
OBJECTS=hodgkin-huxley-cuda-neuronet.cu
all: hh-cuda

hh-cuda: $(OBJECTS) hodgkin-huxley-cuda-neuronet.h
#	nvcc -arch sm_21 -Xcompiler -Wall -O3 --use_fast_math -DOSCILL_SAVE hodgkin-huxley-cuda-neuronet.cu -o hh-cuda
#	nvcc -arch sm_21 -Xcompiler -Wall -O3 --use_fast_math -Xcompiler  $(OBJECTS) -o $@
	nvcc -arch sm_21 -Xcompiler -Wall -O3 --use_fast_math -Xcompiler -pthread -DWITH_MPI $(MPI_INC) $(OBJECTS) $(MPI_LIB) $(MPI_LIB_PATH) -o $@

clean:
	rm hh-cuda