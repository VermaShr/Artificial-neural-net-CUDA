
NVCC        = nvcc

NVCC_FLAGS  = -I/usr/local/cuda/include -std=c++11 -gencode=arch=compute_50,code=\"sm_50,compute_50\"
ifdef dbg
	NVCC_FLAGS  += -g -G
else
	NVCC_FLAGS  += -O2
endif

LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = neuralnet
OBJ	        = neuralnet_cu.o neuralnet_cpp.o

default: $(EXE)

neuralnet_cu.o: neuralnet.cu neuralnet_kernel.cu neuralnet.h
	$(NVCC) -c -o $@ neuralnet.cu $(NVCC_FLAGS)

neuralnet_cpp.o: neuralnet_gold.cpp
	$(NVCC) -c -o $@ neuralnet_gold.cpp $(NVCC_FLAGS) 

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS) $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
