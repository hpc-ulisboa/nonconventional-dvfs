#include "hip/hip_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h> 
#include <unistd.h> 
#include <sys/wait.h>
#include <signal.h>

#include "lcutil.h"

#define TEST_RUN

#define COMP_ITERATIONS (4096) //512
#define THREADS (1024)
#define BLOCKS (32760)
#define N (10)

#define UNROLL_ITERATIONS (32)
#define THREADS_WARMUP (1024)
#define deviceNum (0)

#define OPS 1000


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
			r0 = r1;//r0;
			r1 = r2;//r1;
			r2 = r3;//r2;
			r3 = r0;//r3;
		}
	}
	shared[threadIdx.x] = r0;
}

template <class T> __global__ void benchmark(T* cdin, T* cdout){

	const long ite = blockIdx.x * THREADS + threadIdx.x;

	T r0, r1;

	volatile int m;

	for (int k=0; k<N;k++){
		for(int j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS){
			#pragma unroll
			for(int i=0; i<UNROLL_ITERATIONS; i++){
				r0 = cdin[ite];
				#pragma unroll
				for (int m = 0; m < OPS; ++m)
					asm volatile ("");
				cdout[ite]=r0;
			}
		}
	}
	cdout[ite]=r0;
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

void runbench(double* kernel_time, double* flops, int * hostIn, int * hostOut) {

	hipEvent_t start, stop;
	dim3 dimBlock(THREADS, 1, 1);
	dim3 dimGrid(BLOCKS, 1, 1);

	initializeEvents(&start, &stop);

    hipLaunchKernelGGL((benchmark<int>), dim3(dimGrid), dim3(dimBlock), 0, 0, (int *) hostIn, (int *) hostOut);

	hipDeviceSynchronize();

	double time = finalizeEvents(start, stop);
	
	*kernel_time = time;
}

int main(int argc, char *argv[]){

	int i;
	int device = 0;
	int status;

	#ifdef TEST_RUN
		printf("TEST_RUN\n");
		// Resets the DVFS Settings to guarantee correct MemCpy of the data
		status = system("rocm-smi -r");
		status = system("./DVFS -P 7");
		status = system("./DVFS -p 3");
	#endif

	HIP_SAFE_CALL( hipDeviceReset());
	// Synchronize in order to wait for memory operations to finish
	HIP_SAFE_CALL(hipDeviceSynchronize());

	hipDeviceProp_t deviceProp;

	int ntries = 1;
	unsigned int sizeB, size; 
	if(argc > 1) {
		printf("Usage: %s [ntries]\n", argv[0]);
		exit(1);
	}

	// Computes the total sizeB in bits
	size = THREADS*BLOCKS;
	sizeB *= sizeof(int);

	int pid = fork();
	if(pid == 0) {
		char *args[4];
		std::string gpowerSAMPLER = "gpowerSAMPLER_peak";
		std::string e = "-e";
		std::string time_string = "-s 10";
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


		printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
		printf("Buffer sizeB: %luMB\n", size*sizeof(int)/(1024*1024));
		// Initialize Host Memory
		int *hostIn = (int *) malloc(size*sizeof(int));
		int *hostOut = (int *) malloc(size*sizeof(int));
		int *hostOutverification = (int *) calloc(size, sizeof(int));

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

		printf("Malloc\n");
		// Initialize Device Memory
		int *deviceIn, *deviceOut, *deviceOutVeri;
		HIP_SAFE_CALL(hipMalloc((void**)&deviceIn, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceOut, size * sizeof(int)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceOutVeri, size * sizeof(int)));
		// Synchronize in order to wait for memory operations to finish
		HIP_SAFE_CALL(hipDeviceSynchronize());
		printf("MemCpy\n");
		// Transfer data from host to device
		HIP_SAFE_CALL(hipMemcpy(deviceIn, hostIn, size*sizeof(int), hipMemcpyHostToDevice));
		// Synchronize in order to wait for memory operations to finish
		HIP_SAFE_CALL(hipDeviceSynchronize());

		// Run the warmup kernel to flush caches
		//runbench_warmup();
		
		// Synchronize in order to wait for memory operations to finish
		HIP_SAFE_CALL(hipDeviceSynchronize());

		#ifdef TEST_RUN
			printf("applyDVFS\n");
			// Apply custom DVFS profile
			status = system("python applyDVFS.py 7 3");
			printf("Apply DVFS status: %d\n", status);
		#endif

		if(status == 0) {
			printf("Start Testing\n");
			kill(pid, SIGUSR1);
			runbench(&n_time[0][0],&value[0][0], deviceIn, deviceOut);
			kill(pid, SIGUSR2);
			printf("End Testing\n");
			printf("Registered time: %f ms\n", n_time[0][0]);

			// Resets the DVFS Settings
			int status = system("rocm-smi -r");
			#ifdef TEST_RUN
				status = system("./DVFS -P 7");
				status = system("./DVFS -p 3");
			#endif
			
			HIP_SAFE_CALL(hipDeviceSynchronize());

			// Transfer data from device to host
			HIP_SAFE_CALL(hipMemcpy(hostOut, deviceOut,  size*sizeof(int), hipMemcpyDeviceToHost));

			// Synchronize in order to wait for memory operations to finish
			HIP_SAFE_CALL(hipDeviceSynchronize());

			// Verification RUN
			printf("Start Testing Verification\n");
			double n_time_verification[ntries][2], value_verification[ntries][4];
			runbench(&n_time_verification[0][0],&value_verification[0][0], deviceIn, deviceOutVeri);
			printf("End Testing Verification\n");
			printf("Registered time DEFAULT DVFS: %f ms\n", n_time_verification[0][0]);


			HIP_SAFE_CALL(hipDeviceSynchronize());

			// Transfer data from device to host
			HIP_SAFE_CALL(hipMemcpy(hostOutverification, deviceOutVeri,  size*sizeof(int), hipMemcpyDeviceToHost));

			// Synchronize in order to wait for memory operations to finish
			HIP_SAFE_CALL(hipDeviceSynchronize());

			// Verification of data transfer
			int errors = 0;
			for (i = 0; i < size; i++) {
				//printf("%d %d\n", hostIn[i], hostOut[i]);
				if(hostOutverification[i] != hostOut[i])
					errors++;
			}
			if(errors == 0) 
				printf("Result: True .\n");
			else {
				printf("Result: False .\n");
				printf("Errors: %d .\n", errors);
			}
		}
		else {
			// Kills gpowerSAMPLER child process
			kill(pid, SIGKILL);

			HIP_SAFE_CALL( hipDeviceReset());
		    free(hostIn);
		    free(hostOut);

		    // Wait for child process to finish
		    pid = wait(&status);

		    return -1;
		}

		HIP_SAFE_CALL(hipDeviceReset());
		free(hostIn);
		free(hostOut);

		// Wait for child process to finish
		pid = wait(&status);
	}
	return 0;
}
