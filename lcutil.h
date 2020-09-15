/**
 * lcutil.h: This file is part of the mixbench GPU micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#ifndef _CUTIL_H_
#define _CUTIL_H_

#include <stdio.h>
#include "hip/hip_runtime.h"

#define HIP_SAFE_CALL( call) {                                    \
    hipError_t  err = call;                                                    \
    if( hipSuccess != err) {                                                \
        fprintf(stderr, "HIP error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, hipGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } }

#define FRACTION_CEILING(numerator, denominator) ((numerator+denominator-1)/(denominator))

#endif
