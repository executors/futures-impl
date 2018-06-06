CXX=g++-7
NVCC=~/install/nvidia/cuda-9.2.88/bin/nvcc

ifdef DEBUG
	FLAGS+=-O0 -g
else
	FLAGS+=-O3 
endif

FLAGS+=-std=c++17
NVCC_FLAGS+=-gencode arch=compute_61,\"code=sm_61\" -gencode arch=compute_61,\"code=compute_61\"
CXX_LIBS=-lpthread
INCLUDES=-I.
CU_SOURCES=$(shell ls -1 *.cpp)
CXX_SOURCES=$(shell ls -1 *.cpp)
PROGRAMS=$(CXX_SOURCES:.cpp=) $(CU_SOURCES:.cu=)
DIRECTORIES=build

all: directories $(PROGRAMS)

.PHONY: directories
directories: $(DIRECTORIES)/  

$(DIRECTORIES)/:
	mkdir -p $@ 

% : %.cpp
	$(CXX) $(FLAGS) $(CXX_LIBS) $(INCLUDES) $< -o build/$@
	build/$@

% : %.cu
	$(NVCC) $(FLAGS) $(NVCC_FLAGS) $(INCLUDES) $< -o build/$@
	build/$@

clean:
	rm -f $(DIRECTORIES)/*
	rmdir $(DIRECTORIES)

