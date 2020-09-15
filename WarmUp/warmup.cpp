#include "hip/hip_runtime.h"
/*
 * Copyright 2011-2015 NVIDIA Corporation. All rights reserved
 *
 * Sample app to demonstrate use of CUPTI library to obtain metric values
 * using callbacks for CUDA runtime APIs
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include "../lcutil.h"

#define COMP_ITERATIONS (16384)
#define THREADS (1024)
#define BLOCKS (32760)
#define REGBLOCK_SIZE (4)
#define UNROLL_ITERATIONS (32)
#define deviceNum (0)


//CODE


template <class T> __global__ void benchmark (int aux){

	__shared__ T shared[THREADS];

	register T r0 = 1.0,
	  r1 = r0+(T)(31),
	  r2 = r0+(T)(37),
	  r3 = r0+(T)(41);

	for(int j=0; j<COMP_ITERATIONS; j+=UNROLL_ITERATIONS){
		#pragma unroll
		for(int i=0; i<UNROLL_ITERATIONS; i++){
			// Each iteration maps to floating point 8 operations (4 multiplies + 4 additions)
			r0 = r1;//r0;
			r1 = r2;//r1;
			r2 = r3;//r2;
			r3 = r0;//r3;
		}
	}
	shared[threadIdx.x] = r0;
	
}

void initializeEvents(hipEvent_t *start, hipEvent_t *stop){
	HIP_SAFE_CALL( hipEventCreate(start) );
	HIP_SAFE_CALL( hipEventCreate(stop) );
	HIP_SAFE_CALL( hipEventRecord(*start, 0) );
}

float finalizeEvents(hipEvent_t start, hipEvent_t stop){
	HIP_SAFE_CALL( hipGetLastError() );
	HIP_SAFE_CALL( hipEventRecord(stop, 0) );
	HIP_SAFE_CALL( hipEventSynchronize(stop) );
	float kernel_time;
	HIP_SAFE_CALL( hipEventElapsedTime(&kernel_time, start, stop) );
	HIP_SAFE_CALL( hipEventDestroy(start) );
	HIP_SAFE_CALL( hipEventDestroy(stop) );
	return kernel_time;
}


void runbench(double* kernel_time){
    int aux=0;
	const long long shared_access = 2*(long long)(COMP_ITERATIONS)*THREADS*BLOCKS;

	dim3 dimBlock(THREADS, 1, 1);
    dim3 dimGrid(BLOCKS, 1, 1);
	hipEvent_t start, stop;

	initializeEvents(&start, &stop);
	
	hipLaunchKernelGGL((benchmark<float>), dim3(dimGrid), dim3(dimBlock ), 0, 0, aux);
	
	hipDeviceSynchronize();

	double time = finalizeEvents(start, stop);
	
	*kernel_time = time;
}


int main(int argc, char *argv[]){
	// CUpti_SubscriberHandle subscriber;
	int device = 0;
	hipDeviceProp_t deviceProp;


	printf("Usage: %s [number of tries]\n", argv[0]);
	int ntries;
	if (argc>1){
		ntries = atoi(argv[1]);
	}else{
		ntries = 1;
	}

	HIP_SAFE_CALL(hipSetDevice(deviceNum));
	double time[ntries][2],value[ntries][4];

	HIP_SAFE_CALL(hipGetDeviceProperties(&deviceProp, device));

	for (int i = 0; i < ntries; i++){
		runbench(&time[0][0]);
		printf("Registered time: %f ms\n",time[0][0]);
	}

	HIP_SAFE_CALL( hipDeviceReset());

	return 0;
}
