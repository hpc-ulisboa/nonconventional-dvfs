#include "hip/hip_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h> 
#include <unistd.h> 
#include <sys/wait.h>
#include <signal.h>

#include "lcutil.h"

#define THREADS_WARMUP (1024)
#define UNROLL_ITERATIONS (32)
#define COMP_ITERATIONS_WARMUP (512) //512

#define THREADS (128)
#define N (4)
#define BLOCKS (256000000/( THREADS * N))

#define SIZE_IN (THREADS * N * BLOCKS)
#define SIZE_OUT (BLOCKS)

#define EXECS 100


#define MIN_NUMBER 0.000001
#define MAX_NUMBER 0.00001
#define PRECISION 1/10000

#define deviceNum (0)

#define INT
//#define FLOAT
//#define DOUBLE

//CODE
__global__ void warmup(int aux){

	__shared__ float shared[THREADS_WARMUP];

	short r0 = 1.0,
		  r1 = r0+(short)(31),
		  r2 = r0+(short)(37),
		  r3 = r0+(short)(41);

	for(int j=0; j<COMP_ITERATIONS_WARMUP; j+=UNROLL_ITERATIONS){
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

template <class T> __global__ void benchmark(T* idata, T* odata){

	__shared__ T shared_data[THREADS];

	unsigned int i, k, tid = threadIdx.x;
	unsigned int index = blockIdx.x*blockDim.x*N+ threadIdx.x;

	// each thread loads multiple elements from global to shared memory
	shared_data[tid] = 0;
	for (i=0; i< N; i++, index += blockDim.x)
		shared_data[tid] += idata[index];
	__syncthreads();

	// do reduction in shared memory
	if(tid < 64)
		shared_data[tid] += shared_data[tid + 64]; __syncthreads(); 
	
	if(tid <32){
		shared_data[tid] += shared_data[tid + 32];
		shared_data[tid] += shared_data[tid + 16];
		shared_data[tid] += shared_data[tid + 8];
		shared_data[tid] += shared_data[tid + 4];
		shared_data[tid] += shared_data[tid + 2];
		shared_data[tid] += shared_data[tid + 1];
	}

	// write result for this block to global mem
	if(tid == 0) 
		odata[blockIdx.x] = shared_data[0];
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

#ifdef INT
void runbench(double* kernel_time, double* flops, int * hostIn, int * hostOut) {

	hipEvent_t start, stop;
	dim3 dimBlock(THREADS, 1, 1);
	dim3 dimGrid(BLOCKS, 1, 1);

	initializeEvents(&start, &stop);

	for (int i = 0; i < EXECS; ++i)	{
		hipLaunchKernelGGL((benchmark<int>), dim3(dimGrid), dim3(dimBlock), 0, 0, (int *) hostIn, (int *) hostOut);
		hipDeviceSynchronize();
	}
    

	double time = finalizeEvents(start, stop);
	
	*kernel_time = time;
}
#endif
#ifdef FLOAT
void runbench(double* kernel_time, double* flops, float * hostIn, float * hostOut) {

	hipEvent_t start, stop;
	dim3 dimBlock(THREADS, 1, 1);
	dim3 dimGrid(BLOCKS, 1, 1);

	initializeEvents(&start, &stop);

	for (int i = 0; i < EXECS; ++i) {
	    hipLaunchKernelGGL((benchmark<float>), dim3(dimGrid), dim3(dimBlock), 0, 0, (float *) hostIn, (float *) hostOut);

		hipDeviceSynchronize();
	}
	double time = finalizeEvents(start, stop);
	
	*kernel_time = time;
}
#endif
#ifdef DOUBLE
void runbench(double* kernel_time, double* flops, double * hostIn, double * hostOut) {

	hipEvent_t start, stop;
	dim3 dimBlock(THREADS, 1, 1);
	dim3 dimGrid(BLOCKS, 1, 1);

	initializeEvents(&start, &stop);

    for (int i = 0; i < EXECS; ++i) {
	    hipLaunchKernelGGL((benchmark<double>), dim3(dimGrid), dim3(dimBlock), 0, 0, (double *) hostIn, (double *) hostOut);

		hipDeviceSynchronize();
	}
	double time = finalizeEvents(start, stop);
	
	*kernel_time = time;
}
#endif


int main(int argc, char *argv[]){

	int i;
	int device = 0;

	hipDeviceProp_t deviceProp;

	int ntries = 1;
	if (argc > 1) {
		printf("Usage: %s \n", argv[0]);
		exit(1);
	}
	unsigned int sizeB;
	// Resets the DVFS Settings
	int status = system("rocm-smi -r");
	status = system("./DVFS -P 7");
	status = system("./DVFS -p 3");

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

		#ifdef INT
			sizeB = SIZE_IN * (int) sizeof(int);
		#endif
		#ifdef FLOAT
			sizeB = SIZE_IN * (int) sizeof(float);
		#endif
		#ifdef DOUBLE
			sizeB = SIZE_IN * (int) sizeof(double);
		#endif
		

		hipSetDevice(deviceNum);

		double n_time[ntries][2], value[ntries][4];

		// DRAM Memory Capacity
		size_t freeCUDAMem, totalCUDAMem;
		HIP_SAFE_CALL(hipMemGetInfo(&freeCUDAMem, &totalCUDAMem));

		HIP_SAFE_CALL(hipGetDeviceProperties(&deviceProp, device));

		// Generates array of random numbers
	    srand((unsigned) time(NULL));

		printf("Total GPU memory %lu, free %lu\n", totalCUDAMem, freeCUDAMem);
		
		// Initialize Host Memory
		#ifdef INT
			printf("Buffer sizeB: %luMB\n", SIZE_IN*sizeof(int)/(1024*1024));
			int *hostIn = (int *) malloc(sizeB);
			int *hostOut = (int *) calloc(SIZE_OUT, sizeof(int *));
			int *defaultOut = (int *) calloc(SIZE_OUT, sizeof(int *));
			int random = 0;
		#endif
		#ifdef FLOAT
			printf("Buffer sizeB: %luMB\n", SIZE_IN*sizeof(float)/(1024*1024));
			float *hostIn = (float *) malloc(sizeB);
			float *hostOut = (float *) calloc(SIZE_OUT, sizeof(float *));
			float *defaultOut = (float *) calloc(SIZE_OUT, sizeof(float *));
			float random = 0;
		#endif
		#ifdef DOUBLE
			printf("Buffer sizeB: %luMB\n", SIZE_IN*sizeof(double)/(1024*1024));
			double *hostIn = (double *) malloc(sizeB);
			double *hostOut = (double *) calloc(SIZE_OUT, sizeof(double *));
			double *defaultOut = (double *) calloc(SIZE_OUT, sizeof(double *));
			double random = 0;
		#endif
		

		// Initialize the input data
		for (i = 0; i < SIZE_IN; i++) {
			#ifdef INT
				random = MIN_NUMBER + static_cast <int> (rand()) /( static_cast <int> (RAND_MAX/(MAX_NUMBER-MIN_NUMBER)));
			#endif
			#ifdef FLOAT
				random = MIN_NUMBER + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(MAX_NUMBER-MIN_NUMBER)));
			#endif
			#ifdef DOUBLE
				random = MIN_NUMBER + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(MAX_NUMBER-MIN_NUMBER)));
			#endif
			//hostIn[i] = random;
			hostIn[i] = random;
		}

		// Initialize Host Memory
		#ifdef INT
			int *deviceIn;
			int *deviceOut;
			HIP_SAFE_CALL(hipMalloc((void**)&deviceIn, SIZE_IN * sizeof(int)));
			HIP_SAFE_CALL(hipMalloc((void**)&deviceOut, SIZE_OUT * sizeof(int)));

			// Synchronize in order to wait for memory operations to finish
			HIP_SAFE_CALL(hipDeviceSynchronize());
			
			// Transfer data from host to device
			HIP_SAFE_CALL(hipMemcpy(deviceIn, hostIn, SIZE_IN*sizeof(int), hipMemcpyHostToDevice));
		#endif
		#ifdef FLOAT
			float *deviceIn;
			float *deviceOut;
			HIP_SAFE_CALL(hipMalloc((void**)&deviceIn, SIZE_IN * sizeof(float)));
			HIP_SAFE_CALL(hipMalloc((void**)&deviceOut, SIZE_OUT * sizeof(float)));

			// Synchronize in order to wait for memory operations to finish
			HIP_SAFE_CALL(hipDeviceSynchronize());
			
			// Transfer data from host to device
			HIP_SAFE_CALL(hipMemcpy(deviceIn, hostIn, SIZE_IN*sizeof(float), hipMemcpyHostToDevice));
		#endif
		#ifdef DOUBLE
			double *deviceIn;
			double *deviceOut;
			HIP_SAFE_CALL(hipMalloc((void**)&deviceIn, SIZE_IN * sizeof(double)));
			HIP_SAFE_CALL(hipMalloc((void**)&deviceOut, SIZE_OUT * sizeof(double)));

			// Synchronize in order to wait for memory operations to finish
			HIP_SAFE_CALL(hipDeviceSynchronize());
			
			// Transfer data from host to device
			HIP_SAFE_CALL(hipMemcpy(deviceIn, hostIn, SIZE_IN*sizeof(double), hipMemcpyHostToDevice));
		#endif
		
		// Synchronize in order to wait for memory operations to finish
		HIP_SAFE_CALL(hipDeviceSynchronize());

		// Synchronize in order to wait for memory operations to finish
		HIP_SAFE_CALL(hipDeviceSynchronize());

		int status = system("python applyDVFS.py 7 3");
		printf("Apply DVFS status: %d\n", status);

		if(status == 0) {

			printf("Start Testing\n");
			kill(pid, SIGUSR1);
			runbench(&n_time[0][0],&value[0][0], deviceIn, deviceOut);
			kill(pid, SIGUSR2);
			printf("End Testing\n");
			printf("Registered time: %f ms\n", n_time[0][0]);

			// Resets the DVFS Settings
			int status = system("rocm-smi -r");
			status = system("./DVFS -P 7");
			status = system("./DVFS -p 3");

		
			HIP_SAFE_CALL(hipDeviceSynchronize());

			// Transfer data from device to host
			#ifdef INT
				HIP_SAFE_CALL(hipMemcpy(hostOut, deviceOut,  SIZE_OUT*sizeof(int), hipMemcpyDeviceToHost));
			#endif
			#ifdef FLOAT
				HIP_SAFE_CALL(hipMemcpy(hostOut, deviceOut,  SIZE_OUT*sizeof(float), hipMemcpyDeviceToHost));
			#endif
			#ifdef DOUBLE
				HIP_SAFE_CALL(hipMemcpy(hostOut, deviceOut,  SIZE_OUT*sizeof(double), hipMemcpyDeviceToHost));
			#endif

			// Synchronize in order to wait for memory operations to finish
			HIP_SAFE_CALL(hipDeviceSynchronize());

			runbench_warmup();
			
			// Rerun  the kernel using conventional DVFS settings
			double n_time_default[ntries][2], value_default[ntries][4];
			runbench(&n_time_default[0][0],&value_default[0][0], deviceIn, deviceOut);
			printf("Registered time DEFAULT DVFS: %f ms\n", n_time_default[0][0]);

			// Synchronize in order to wait for memory operations to finish
			HIP_SAFE_CALL(hipDeviceSynchronize());

			// Transfer data from device to host
			#ifdef INT
				HIP_SAFE_CALL(hipMemcpy(defaultOut, deviceOut,  SIZE_OUT*sizeof(int), hipMemcpyDeviceToHost));
			#endif
			#ifdef FLOAT
				HIP_SAFE_CALL(hipMemcpy(defaultOut, deviceOut,  SIZE_OUT*sizeof(float), hipMemcpyDeviceToHost));
			#endif
			#ifdef DOUBLE
				HIP_SAFE_CALL(hipMemcpy(defaultOut, deviceOut,  SIZE_OUT*sizeof(double), hipMemcpyDeviceToHost));
			#endif
			
			// Synchronize in order to wait for memory operations to finish
			HIP_SAFE_CALL(hipDeviceSynchronize());

			// Verification of output
			int failed = 0;
			double abss = -2.0f, valueAbs;
			for (i = 0; i < SIZE_OUT; i++) {
				/*#ifdef INT
					printf("%d %d\n", defaultOut[i], hostOut[i]);
				#endif
				#ifdef FLOAT
					printf("%f %f\n", defaultOut[i], hostOut[i]);
				#endif
				#ifdef DOUBLE
					printf("%lf %lf\n", defaultOut[i], hostOut[i]);
				#endif*/
				valueAbs = abs(defaultOut[i] - hostOut[i]);
				if(valueAbs > abss)
					abss = valueAbs;
				/*if(abss > PRECISION) {
					failed++;
				}*/
				if(defaultOut[i] != hostOut[i]) {
					failed++;
				}
			}
			if(failed == 0) 
				printf("Result: True .\n");
			else {
				printf("Result: False .\n");
				printf("Size: %d Number of failures: %d\n", SIZE_OUT, failed);
				printf("ABS %f\n", abss);
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
