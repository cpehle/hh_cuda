MPI_INC=-I/common/openmpi/include
MPI_LIB_PATH=-L/common/openmpi/lib
MPI_LIB=-lmpi_cxx -lmpi
# MPI_INC=-I/usr/lib/openmpi/include/openmpi -I/usr/lib/openmpi/include
# MPI_LIB_PATH=-L/usr/lib -L/usr/lib/openmpi/lib
# MPI_LIB=-lmpi_cxx -lmpi -ldl -lhwloc
OBJECT=hodgkin-huxley-cuda-neuronet.cu

all: hh-cuda

hh-cuda: $(OBJECT) hodgkin-huxley-cuda-neuronet.h
# 	nvcc -arch sm_21 -Xcompiler -Wall -O3 --use_fast_math $(OBJECT) -o $@
	nvcc -arch sm_21 -Xcompiler -Wall -O3 --use_fast_math -Xcompiler -pthread -DWITH_MPI $(MPI_INC) $(OBJECT) $(MPI_LIB) $(MPI_LIB_PATH) -o $@

debug: $(OBJECTS) hodgkin-huxley-cuda-neuronet.h
	nvcc -arch sm_21 -Xcompiler -g3 -Xcompiler -Wall -O0 $(OBJECT) -o hh-cuda
clean:
	rm hh-cuda
