#include "hip/hip_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h> 
#include <unistd.h> 
#include <sys/wait.h>
#include <signal.h>

#include "lcutil.h"


#define COMP_ITERATIONS (2048) //512
#define REGBLOCK_sizeB (4)
#define UNROLL_ITERATIONS (32)//32
#define THREADS_WARMUP (1024)

#define THREADS (1024)
#define BLOCKS (32760)
#define deviceNum (0)
#define RANGE (15)

#define MIN_NUMBER 0.000001
#define MAX_NUMBER 1.5
#define PRECISION 1/10000

#define TEST_RUN


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

	const long ite=(blockIdx.x * THREADS + threadIdx.x) * 4;
	long j;

	register T r0, r1, r2, r3;

	r0=cdin[ite];
	r1=cdin[ite+1];
	r2=cdin[ite+2];
	r3=cdin[ite+3];

	for(j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS){
		#pragma unroll
		for(int i=0; i<UNROLL_ITERATIONS; i++){
			r0 = exp(r2);
            r1 = cos(r3);
            r2 = log(r0);
            r3 = sin(r1);
		}
	}

	cdout[ite/4]=r0;
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

void runbench(double* kernel_time, double* flops, float * hostIn, float * hostOut) {

	hipEvent_t start, stop;
	dim3 dimBlock(THREADS, 1, 1);
	dim3 dimGrid(BLOCKS, 1, 1);

	initializeEvents(&start, &stop);

    hipLaunchKernelGGL((benchmark<float>), dim3(dimGrid), dim3(dimBlock), 0, 0, (float *) hostIn, (float *) hostOut);

	hipDeviceSynchronize();

	double time = finalizeEvents(start, stop);
	
	*kernel_time = time;
}

int main(int argc, char *argv[]){

	int i;
	int device = 0;
	int status;
	hipDeviceProp_t deviceProp;

	int ntries = 1;
	unsigned int sizeB, size; 
	if (argc > 1) {
		printf("Usage: %s \n", argv[0]);
		exit(1);
	}

	#ifdef TEST_RUN
		printf("TEST_RUN\n");
		// Resets the DVFS Settings
		status = system("rocm-smi -r");
		status = system("./DVFS -P 7");
		status = system("./DVFS -p 3");
	#endif

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

		// Computes the total sizeB in bits
		size = (THREADS * BLOCKS) * 4;
		sizeB = size * (int) sizeof(float);

		hipSetDevice(deviceNum);

		double n_time[ntries][2], value[ntries][4];

		// DRAM Memory Capacity
		size_t freeCUDAMem, totalCUDAMem;
		HIP_SAFE_CALL(hipMemGetInfo(&freeCUDAMem, &totalCUDAMem));

		HIP_SAFE_CALL(hipGetDeviceProperties(&deviceProp, device));

		printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
		printf("Buffer sizeB: %luMB\n", size*sizeof(float)/(1024*1024));
		
		// Initialize Host Memory
		float *hostIn = (float *) malloc(sizeB);
		float *hostOut = (float *) calloc(size/4, sizeof(float *));
		float *defaultOut = (float *) calloc(size/4, sizeof(float *));

		// Generates array of random numbers
	    srand((unsigned) time(NULL));
		float random = 0;
		// Initialize the input data
		for (i = 0; i < size; i++) {
			random = MIN_NUMBER + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(MAX_NUMBER-MIN_NUMBER)));
			hostIn[i] = random;
		}

		// Initialize Host Memory
		float *deviceIn;
		float *deviceOut;
		HIP_SAFE_CALL(hipMalloc((void**)&deviceIn, size * sizeof(float)));
		HIP_SAFE_CALL(hipMalloc((void**)&deviceOut, (size/4) * sizeof(float)));

		// Synchronize in order to wait for memory operations to finish
		HIP_SAFE_CALL(hipDeviceSynchronize());
		
		// Transfer data from host to device
		HIP_SAFE_CALL(hipMemcpy(deviceIn, hostIn, size*sizeof(float), hipMemcpyHostToDevice));
		// Synchronize in order to wait for memory operations to finish
		HIP_SAFE_CALL(hipDeviceSynchronize());

		printf("Start warmup\n");
		for (i=0;i<1;i++){
			runbench_warmup();
		}
		printf("End warmup\n");
		// Synchronize in order to wait for memory operations to finish
		HIP_SAFE_CALL(hipDeviceSynchronize());

		#ifdef TEST_RUN
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
			status = system("rocm-smi -r");
			#ifdef TEST_RUN
				status = system("./DVFS -P 7");
				status = system("./DVFS -p 3");
			#endif
		
			HIP_SAFE_CALL(hipDeviceSynchronize());

			// Transfer data from device to host
			HIP_SAFE_CALL(hipMemcpy(hostOut, deviceOut,  size/4*sizeof(float), hipMemcpyDeviceToHost));

			// Synchronize in order to wait for memory operations to finish
			HIP_SAFE_CALL(hipDeviceSynchronize());

			for (i=0;i<1;i++){
				runbench_warmup();
			}
			// Rerun  the kernel using conventional DVFS settings
			runbench(&n_time[0][0],&value[0][0], deviceIn, deviceOut);
			printf("Registered time DEFAULT DVFS: %f ms\n", n_time[0][0]);

			// Synchronize in order to wait for memory operations to finish
			HIP_SAFE_CALL(hipDeviceSynchronize());

			// Transfer data from device to host
			HIP_SAFE_CALL(hipMemcpy(defaultOut, deviceOut,  size/4*sizeof(float), hipMemcpyDeviceToHost));

			// Synchronize in order to wait for memory operations to finish
			HIP_SAFE_CALL(hipDeviceSynchronize());

			// Verification of output
			int failed = 0;
			for (i = 0; i < size/4; i++) {
				//printf("%f - %f = %f\n", defaultOut[i], hostOut[i], abs(defaultOut[i] - hostOut[i]));
				if(abs(defaultOut[i] - hostOut[i]) > PRECISION) {
					failed++;
				}
			}
			if(failed == 0) 
				printf("Result: True .\n");
			else {
				printf("Result: False .\n");
				printf("Size: %d Number of failures: %d\n", size/4, failed);
			}
		}
		else {
			kill(pid, SIGKILL);

			HIP_SAFE_CALL( hipDeviceReset());

		    free(hostIn);
		    free(hostOut);

		    // Wait for child process t finish
		    pid = wait(&status);

		    return -1;
		}

		HIP_SAFE_CALL(hipDeviceReset());
		free(hostIn);
		free(hostOut);

		pid = wait(&status);
	}
	return 0;
}
