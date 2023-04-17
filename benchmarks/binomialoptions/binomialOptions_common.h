/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef BINOMIALOPTIONS_COMMON_H
#define BINOMIALOPTIONS_COMMON_H

////////////////////////////////////////////////////////////////////////////////
// Global parameters
////////////////////////////////////////////////////////////////////////////////

using real = double;

////////////////////////////////////////////////////////////////////////////////
// Black-Scholes formula for binomial tree results validation
////////////////////////////////////////////////////////////////////////////////

void BlackScholesCall(
    real &callResult,
    const real S, const real X, const real T, const real R, const real V
);


////////////////////////////////////////////////////////////////////////////////
// Process single option on CPU
// Note that CPU code is for correctness testing only and not for benchmarking.
////////////////////////////////////////////////////////////////////////////////

#pragma omp begin declare target
real binomialOptionsOffload(
                            const int NUM_STEPS, double *Call,
    const real S, const real X, const real T, const real R, const real V
);
double expiryCallValue(const int NUM_STEPS, double S, double X, double vDt, int i);
void warmup();
#pragma omp end declare target

void ArrangeInputsThreadLocality(int numOptions, int num_threads, real *&S, real *&X, real *&T, real *&R, real *&V);

#endif
