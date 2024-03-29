#MPI_INC=-I/common/openmpi/include
#MPI_LIB_PATH=-L/common/openmpi/lib
#MPI_LIB=-lmpi_cxx -lmpi
#CUDA_INC=-I/common/cuda-5.5/include/
#CUDA_LIB_PATH=-L/common/cuda-5.5/lib64/
MPI_INC=-I/usr/lib/openmpi/include/openmpi -I/usr/lib/openmpi/include
MPI_LIB_PATH=-L/usr/lib -L/usr/lib/openmpi/lib
MPI_LIB=-lmpi_cxx -lmpi -ldl -lhwloc
CUDA_INC=-I/usr/local/cuda/include
CUDA_LIB_PATH=-L/usr/local/cuda/lib64/

OBJECT=hodgkin-huxley-cuda-neuronet.cpp

all: hh-cuda

hh-cuda: $(OBJECT) hodgkin-huxley-cuda-neuronet.h hh-kernels.h hh-kernels.o common_declarations.h
#	nvcc -arch sm_21 -Xcompiler -Wall -O3 --use_fast_math hh-kernels.o $(OBJECT) -o $@
	nvcc -arch sm_21 -Xcompiler -Wall -O3 --use_fast_math -Xcompiler -pthread -DWITH_MPI -DOSCILL_SAVE $(MPI_INC) hh-kernels.o  $(OBJECT) $(MPI_LIB) $(MPI_LIB_PATH) -o $@
#	g++ -O3 -pthread -DWITH_MPI $(MPI_INC) $(CUDA_INC) $(CUDA_LIB_PATH) $(MPI_LIB_PATH) hh-kernels.o $(OBJECT) $(MPI_LIB) -lcudart_static -ldl -lpthread -lrt -o $@
#	g++ -O3 -pthread $(CUDA_INC) $(CUDA_LIB_PATH) hh-kernels.o $(OBJECT) -lcudart_static -ldl -lpthread -lrt -o $@

hh-kernels.o: hh-kernels.cu common_declarations.h
	nvcc -c -arch sm_21 -Xcompiler -Wall -O3 --use_fast_math -DOSCILL_SAVE hh-kernels.cu -o $@

clean:
	rm hh-cuda hh-kernels.o
