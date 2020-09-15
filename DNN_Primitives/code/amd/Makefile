SOURCE_DIR?=.
BIN_DIR?=bin
MKDIR=mkdir -p

#hipcc
HIPCC=/opt/rocm/hip/bin/hipcc

#ROCm include directory
ROCM_INCLUDE=/opt/rocm/include

#BLAS
ROCBLAS_LIB=rocblas

#CONV
MIOPEN_LIB?=MIOpen

#DeepBench
DEEPBENCH_INC=${SOURCE_DIR}/../kernels

all: conv rnn gemm 

#OPT=-g -O0 -fsanitize=undefined -fno-omit-frame-pointer
OPT=-O3

conv:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/conv_bench_rocm.cpp -o $(BIN_DIR)/conv_bench -I$(DEEPBENCH_INC) -I$(ROCM_INCLUDE) -l$(MIOPEN_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx900

rnn:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/rnn_bench_rocm.cpp -o $(BIN_DIR)/rnn_bench -I$(DEEPBENCH_INC) -I$(ROCM_INCLUDE) -l$(MIOPEN_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx900

gemm:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/gemm_bench.cpp -o $(BIN_DIR)/gemm_bench -I$(DEEPBENCH_INC) -I$(ROCM_INCLUDE) -l$(ROCBLAS_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx900

eventGEMM:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/eventGEMM.cpp -o $(BIN_DIR)/eventGEMM -I$(DEEPBENCH_INC) -I$(ROCM_INCLUDE) -l$(ROCBLAS_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx900

eventConv:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/eventConv.cpp -o $(BIN_DIR)/eventConv -I$(DEEPBENCH_INC) -I$(ROCM_INCLUDE) -l$(MIOPEN_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx900

eventConvSelect:
	$(MKDIR) $(BIN_DIR)
	$(HIPCC) ${SOURCE_DIR}/eventConvSelect.cpp -o $(BIN_DIR)/eventConvSelect -I$(DEEPBENCH_INC) -I$(ROCM_INCLUDE) -l$(MIOPEN_LIB) $(OPT) -std=c++11 --amdgpu-target=gfx900


clean:
	rm -rf $(BIN_DIR)
