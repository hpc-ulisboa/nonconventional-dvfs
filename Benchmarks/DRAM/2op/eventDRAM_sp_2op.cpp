#include "hip/hip_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h> 
#include <unistd.h> 
#include <sys/wait.h>
#include <signal.h>

#include "../../../lcutil.h"


#define COMP_ITERATIONS (4096) //512
#define REGBLOCK_sizeB (4)
#define UNROLL_ITERATIONS (32)
#define THREADS_WARMUP (1024)
#define THREADS (1024)
int BLOCKS;
#define deviceNum (0)


//CODE
__global__ void warmup(int aux){

	__shared__ double shared[THREADS_WARMUP];

	short r0 = 1.0,
		  r1 = r0+(short)(31),
		  r2 = r0+(short)(37),
		  r3 = r0+(short)(41);

	for(int j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS){
		#pragma unroll
		for(int i=0; i<UNROLL_ITERATIONS; i++){
			// Each iteration maps to doubleing point 8 operations (4 multiplies + 4 additions)
			r0 = r0 * r0 + r1;//r0;
			r1 = r1 * r1 + r2;//r1;
			r2 = r2 * r2 + r3;//r2;
			r3 = r3 * r3 + r0;//r3;
		}
	}
	shared[threadIdx.x] = r0;
}

template <class T> __global__ void benchmark(T* cdin0, T* cdin1, T* cdin2, T* cdin3, T* cdin4, T* cdin5, T* cdin6, T* cdin7, T* cdin8, T* cdin9, 
											 T* cdout0, T* cdout1, T* cdout2, T* cdout3, T* cdout4, T* cdout5, T* cdout6, T* cdout7, T* cdout8, T* cdout9){

	const long ite=blockIdx.x * THREADS + threadIdx.x;

	register T  r0, r1, r2;

	r0=cdin0[ite];
	r1 = r0 + 23;
	r2 = r0 + r1 + r0;
	cdout0[ite]=r0;

	r0=cdin1[ite];
	r1 = r0 + r2;
	r2 = r0 + r1 + r0;
	cdout1[ite]=r0;

	r0=cdin2[ite];
	r1 = r0 + r2;
	r2 = r0 + r1 + r0;
	cdout2[ite]=r0;

	r0=cdin3[ite];
	r1 = r0 + r2;
	r2 = r0 + r1 + r0;
	cdout3[ite]=r0;

	r0=cdin4[ite];
	r1 = r0 + r2;
	r2 = r0 + r1 + r0;
	cdout4[ite]=r0;

	r0=cdin5[ite];
	r1 = r0 + r2;
	r2 = r0 + r1 + r0;
	cdout5[ite]=r0;

	r0=cdin6[ite];
	r1 = r0 + r2;
	r2 = r0 + r1 + r0;
	cdout6[ite]=r0;

	r0=cdin7[ite];
	r1 = r0 + r2;
	r2 = r0 + r1 + r0;
	cdout7[ite]=r0;

	r0=cdin8[ite];
	r1 = r0 + r2;
	r2 = r0 + r1 + r0;
	cdout8[ite]=r0;

	r0=cdin9[ite];
	r1 = r0 + r2;
	r2 = r0 + r1 + r0;
	cdout9[ite]=r0;
}

void initializeEvents(hipEvent_t *start, hipEvent_t *stop){
	HIP_SAFE_CALL( hipEventCreate(start) );
	HIP_SAFE_CALL( hipEventCreate(stop) );
	HIP_SAFE_CALL( hipEventRecord(*start, 0) );
}

double finalizeEvents(hipEvent_t start, hipEvent_t stop){
	HIP_SAFE_CALL( hipGetLastError() );
	HIP_SAFE_CALL( hipEventRecord(stop, 0) );
	HIP_SAFE_CALL( hipEventSynchronize(stop) );
	float kernel_time;
	HIP_SAFE_CALL( hipEventElapsedTime(&kernel_time, start, stop) );
	HIP_SAFE_CALL( hipEventDestroy(start) );
	HIP_SAFE_CALL( hipEventDestroy(stop) );
	return kernel_time;
}

void runbench_warmup(){
	const int BLOCK_sizeB = 256;
	const int TOTAL_REDUCED_BLOCKS = 256;
    int aux=0;

	dim3 dimBlock(BLOCK_sizeB, 1, 1);
	dim3 dimReducedGrid(TOTAL_REDUCED_BLOCKS, 1, 1);

	hipLaunchKernelGGL((warmup), dim3(dimReducedGrid), dim3(dimBlock ), 0, 0, aux);
	HIP_SAFE_CALL( hipGetLastError() );
	HIP_SAFE_CALL( hipDeviceSynchronize() );
}

void runbench(double* kernel_time, double* flops, int * hostIn0, int * hostIn1, int * hostIn2, int * hostIn3, int * hostIn4, int * hostIn5, int * hostIn6, int * hostIn7, int * hostIn8, int * hostIn9, 
    											  int * hostOut0, int * hostOut1, int * hostOut2, int * hostOut3, int * hostOut4, int * hostOut5, int * hostOut6, int * hostOut7, int * hostOut8, int * hostOut9) {

	hipEvent_t start, stop;
	dim3 dimBlock(THREADS, 1, 1);
	dim3 dimGrid(BLOCKS, 1, 1);

	initializeEvents(&start, &stop);

    hipLaunchKernelGGL((benchmark<int>), dim3(dimGrid), dim3(dimBlock), 0, 0, (int *) hostIn0, (int *) hostIn1, (int *) hostIn2, (int *) hostIn3, (int *) hostIn4, (int *) hostIn5, (int *) hostIn6, (int *) hostIn7, (int *) hostIn8, (int *) hostIn9, 
    																		  (int *) hostOut0, (int *) hostOut1, (int *) hostOut2, (int *) hostOut3, (int *) hostOut4, (int *) hostOut5, (int *) hostOut6, (int *) hostOut7, (int *) hostOut8, (int *) hostOut9);

	hipDeviceSynchronize();

	double time = finalizeEvents(start, stop);
	
	*kernel_time = time;
}

int main(int argc, char *argv[]){

	int i;
	int device = 0;

	// Resets the DVFS Settings to guarantee correct MemCpy of the data
	int status = system("rocm-smi -r");
	status = system("./DVFS -P 7");
	status = system("./DVFS -p 3");

	// Synchronize in order to wait for memory operations to finish
	HIP_SAFE_CALL(hipDeviceSynchronize());

	hipDeviceProp_t deviceProp;

	int ntries;
	unsigned int sizeB, size; 
	if (argc > 2) {
		sizeB = atoi(argv[1]);
		ntries = atoi(argv[2]);
	}
	else if(argc > 1) {
		sizeB = atoi(argv[1]);
		ntries = 1;
	}
	else {
		printf("Usage: %s [buffer sizeB kBytes] [ntries]\n", argv[0]);
		exit(1);
	}

	// Computes the total sizeB in bits
	sizeB *= 1024;
	size = sizeB/(int)sizeof(int);

	if(size % 1024 != 0) {
		printf("sizeB not divisible by 1024!!!\n");
		exit(1);
	}

	// Kernel Config
	BLOCKS = size/1024;

	int pid = fork();
	if(pid == 0) {
		char *args[4];
		std::string gpowerSAMPLER = "gpowerSAMPLER_peak";
		std::string e = "-e";
		std::string time_string = "-s 1";
		args[0] = (char *) gpowerSAMPLER.c_str();
		args[1] = (char *) e.c_str();
		args[2] = (char *) time_string.c_str();
		args[3] = NULL;
		if( execvp(args[0], args) == -1) {
			printf("Error lauching gpowerSAMPLER_peak.\n");
		}
		exit(0);
	}
	else {
		hipSetDevice(deviceNum);

		double n_time[ntries][2], value[ntries][4];

		// DRAM Memory Capacity
		size_t freeCUDAMem, totalCUDAMem;
		HIP_SAFE_CALL(hipMemGetInfo(&freeCUDAMem, &totalCUDAMem));

		HIP_SAFE_CALL(hipGetDeviceProperties(&deviceProp, device));

		printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
		printf("Buffer sizeB: %luMB\n", size*sizeof(int)/(1024*1024));
		
		// Initialize Host Memory
		int *hostIn = (int *) malloc(sizeB);
		int **hostOut = (int **) malloc(10 * sizeof(int*));
		for (int i = 0; i < 10; ++i)
		{
			hostOut[i] = (int *) calloc(size, sizeof(int));
		}

		// Generates array of random numbers
	    srand((unsigned) time(NULL));
	    int sum = 0;
		int random = 0;
		// Initialize the input data
		for (i = 0; i < size-1; i++) {
			random = ((unsigned)rand() << 17) | ((unsigned)rand() << 2) | ((unsigned)rand() & 3);
			hostIn[i] = random;
			sum += random;
		}
		// Places the sum on the last vector position
		hostIn[i] = sum;

		// Initialize Device Memory
		int *deviceIn, *deviceIn1, *deviceIn2, *deviceIn3, *deviceIn4, *deviceIn5, *deviceIn6, *deviceIn7, *deviceIn8, *deviceIn9;
		int *deviceOut, *deviceOut1, *deviceOut2, *deviceOut3, *deviceOut4, *deviceOut5, *deviceOut6, *deviceOut7, *deviceOut8, *deviceOut9;
		HIP_SAFE_CALL(hipMalloc((void**)&deviceIn, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceIn1, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceIn2, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceIn3, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceIn4, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceIn5, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceIn6, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceIn7, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceIn8, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceIn9, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceOut, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceOut1, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceOut2, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceOut3, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceOut4, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceOut5, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceOut6, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceOut7, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceOut8, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceOut9, size * sizeof(int)));
		// Synchronize in order to wait for memory operations to finish
		HIP_SAFE_CALL(hipDeviceSynchronize());
		
		// Transfer data from host to device
		HIP_SAFE_CALL(hipMemcpy(deviceIn, hostIn, size*sizeof(int), hipMemcpyHostToDevice));
		HIP_SAFE_CALL(hipMemcpy(deviceIn1, hostIn, size*sizeof(int), hipMemcpyHostToDevice));
		HIP_SAFE_CALL(hipMemcpy(deviceIn2, hostIn, size*sizeof(int), hipMemcpyHostToDevice));
		HIP_SAFE_CALL(hipMemcpy(deviceIn3, hostIn, size*sizeof(int), hipMemcpyHostToDevice));
		HIP_SAFE_CALL(hipMemcpy(deviceIn4, hostIn, size*sizeof(int), hipMemcpyHostToDevice));
		HIP_SAFE_CALL(hipMemcpy(deviceIn5, hostIn, size*sizeof(int), hipMemcpyHostToDevice));
		HIP_SAFE_CALL(hipMemcpy(deviceIn6, hostIn, size*sizeof(int), hipMemcpyHostToDevice));
		HIP_SAFE_CALL(hipMemcpy(deviceIn7, hostIn, size*sizeof(int), hipMemcpyHostToDevice));
		HIP_SAFE_CALL(hipMemcpy(deviceIn8, hostIn, size*sizeof(int), hipMemcpyHostToDevice));
		HIP_SAFE_CALL(hipMemcpy(deviceIn9, hostIn, size*sizeof(int), hipMemcpyHostToDevice));
		// Synchronize in order to wait for memory operations to finish
		HIP_SAFE_CALL(hipDeviceSynchronize());

		// Run the warmup kernel to flush caches
		for (i=0;i<1;i++){
			runbench_warmup();
		}
		// Synchronize in order to wait for memory operations to finish
		HIP_SAFE_CALL(hipDeviceSynchronize());

		// Apply custom DVFS profile
		int status = system("python applyDVFS.py 7 3");
		printf("Apply DVFS status: %d\n", status);

		if(status == 0) {
			printf("Start Testing\n");
			kill(pid, SIGUSR1);
			runbench(&n_time[0][0],&value[0][0], deviceIn, deviceIn1, deviceIn2, deviceIn3, deviceIn4, deviceIn5, deviceIn6, deviceIn7, deviceIn8, deviceIn9,
												 deviceOut, deviceOut1, deviceOut2, deviceOut3, deviceOut4, deviceOut5, deviceOut6, deviceOut7, deviceOut8, deviceOut9);
			kill(pid, SIGUSR2);
			printf("End Testing\n");
			printf("Registered time: %f ms\n", n_time[0][0]);

			// Resets the DVFS Settings
			int status = system("rocm-smi -r");
			status = system("./DVFS -P 7");
			status = system("./DVFS -p 3");

			HIP_SAFE_CALL(hipDeviceSynchronize());

			// Transfer data from device to host
			HIP_SAFE_CALL(hipMemcpy(hostOut[0], deviceOut,  size*sizeof(int), hipMemcpyDeviceToHost));
			HIP_SAFE_CALL(hipMemcpy(hostOut[1], deviceOut1, size*sizeof(int), hipMemcpyDeviceToHost));
			HIP_SAFE_CALL(hipMemcpy(hostOut[2], deviceOut2, size*sizeof(int), hipMemcpyDeviceToHost));
			HIP_SAFE_CALL(hipMemcpy(hostOut[3], deviceOut3, size*sizeof(int), hipMemcpyDeviceToHost));
			HIP_SAFE_CALL(hipMemcpy(hostOut[4], deviceOut4, size*sizeof(int), hipMemcpyDeviceToHost));
			HIP_SAFE_CALL(hipMemcpy(hostOut[5], deviceOut5, size*sizeof(int), hipMemcpyDeviceToHost));
			HIP_SAFE_CALL(hipMemcpy(hostOut[6], deviceOut6, size*sizeof(int), hipMemcpyDeviceToHost));
			HIP_SAFE_CALL(hipMemcpy(hostOut[7], deviceOut7, size*sizeof(int), hipMemcpyDeviceToHost));
			HIP_SAFE_CALL(hipMemcpy(hostOut[8], deviceOut8, size*sizeof(int), hipMemcpyDeviceToHost));
			HIP_SAFE_CALL(hipMemcpy(hostOut[9], deviceOut9, size*sizeof(int), hipMemcpyDeviceToHost));

			// Synchronize in order to wait for memory operations to finish
			HIP_SAFE_CALL(hipDeviceSynchronize());


			// Verification of data transfer
			int sucess = 0;
			for (int j = 0; j < 10; ++j)
			{
				int sum_received = 0;
				for (i = 0; i < size-1; i++) {
					sum_received += hostOut[j][i];
				}
				if(sum == sum_received && sum == hostOut[j][size-1]) 
					sucess++;
				else
					printf("sum: %d != sum_received: %d\n", sum, sum_received);
			}
			if(sucess == 10)
				printf("Result: True .\n");
			else
				printf("Result: False .\n");
		}
		else {
			// Kills gpowerSAMPLER child process
			kill(pid, SIGKILL);

			HIP_SAFE_CALL( hipDeviceReset());
		    free(hostIn);
			for (int i = 0; i < 10; ++i){
				free(hostOut[i]);
			}
		    free(hostOut);

		    // Wait for child process to finish
		    pid = wait(&status);

		    return -1;
		}

		HIP_SAFE_CALL(hipDeviceReset());
		free(hostIn);
		for (int i = 0; i < 10; ++i){
			free(hostOut[i]);
		}
		free(hostOut);

		// Wait for child process to finish
		pid = wait(&status);
	}
	return 0;
}
