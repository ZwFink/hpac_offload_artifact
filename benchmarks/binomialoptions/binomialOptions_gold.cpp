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

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <algorithm>
#include "binomialOptions_common.h"


///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////

#pragma omp begin declare target
static double CND(double d)
{
    const double       A1 = 0.31938153;
    const double       A2 = -0.356563782;
    const double       A3 = 1.781477937;
    const double       A4 = -1.821255978;
    const double       A5 = 1.330274429;
    const double RSQRT2PI = 0.39894228040143267793994605993438;

    double
    K = 1.0 / (1.0 + 0.2316419 * fabs(d));

    double
    cnd = RSQRT2PI * exp(- 0.5 * d * d) *
          (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if (d > 0)
        cnd = 1.0 - cnd;

    return cnd;
}


void BlackScholesCall(
    real &callResult,
    const real S, const real X,
    const real T, const real R,
    const real V
)
{
    double sqrtT = sqrt(T);

    double    d1 = (log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT);
    double    d2 = d1 - V * sqrtT;

    double CNDD1 = CND(d1);
    double CNDD2 = CND(d2);

    //Calculate Call and Put simultaneously
    double expRT = exp(- R * T);

    callResult   = (real)(S * CNDD1 - X * expRT * CNDD2);
}


////////////////////////////////////////////////////////////////////////////////
// Process an array of OptN options on CPU
// Note that CPU code is for correctness testing only and not for benchmarking.
////////////////////////////////////////////////////////////////////////////////

double expiryCallValue(const int NUM_STEPS, double S, double X, double vDt, int i)
{
    double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
    return (d > 0) ? d : 0;
}

real binomialOptionsOffload(
                            const int NUM_STEPS, double *Call, const real S, const real X, const real T, const real R, const real V

)
{
    const double      dt = T / (double)NUM_STEPS;
    const double     vDt = V * sqrt(dt);
    const double     rDt = R * dt;

    //Per-step interest and discount factors
    const double      If = exp(rDt);
    const double      Df = exp(-rDt);

    //Values and pseudoprobabilities of upward and downward moves
    const double       u = exp(vDt);
    const double       d = exp(-vDt);
    const double      pu = (If - d) / (u - d);
    const double      pd = 1.0 - pu;
    const double  puByDf = pu * Df;
    const double  pdByDf = pd * Df;
    const int nthreads = omp_get_num_teams() * omp_get_num_threads();

    ///////////////////////////////////////////////////////////////////////
    // Compute values at expiration date:
    // call option value at period end is V(T) = S(T) - X
    // if S(T) is greater than X, or zero otherwise.
    // The computation is similar for put options.
    ///////////////////////////////////////////////////////////////////////

    for (int i = 0; i <= NUM_STEPS; i++)
      Call[i*nthreads] = expiryCallValue(NUM_STEPS, S, X, vDt, i);

    ////////////////////////////////////////////////////////////////////////
    // Walk backwards up binomial tree
    ////////////////////////////////////////////////////////////////////////

    for (int i = NUM_STEPS; i > 0; i--)
        for (int j = 0; j <= i - 1; j++)
          Call[j*nthreads] = puByDf * Call[nthreads*(j + 1)] + pdByDf * Call[j*nthreads];

    return (real)Call[0];
}
#pragma omp end declare target

void warmup()
{
  int N = 1024;
  int *A = new int[N];
  int *B = new int[N];
  int *C = new int[N];


  std::fill(B, B+N, 0);
  std::fill(C, C+N, 1);

#pragma omp target data map(tofrom:A[0:N], B[0:N], C[0:N])
  {
  #pragma omp target teams distribute parallel for
  for(int i = 0; i < N; i++)
    {
      A[i] = B[i] + C[i];
    }

  }
  delete[] A;
  delete[] B;
  delete[] C;

}

void ArrangeInputsThreadLocality(int numOptions, int num_threads, real *&S, real *&X, real *&T, real *&R, real *&V)
{
  real *S_n, *X_n, *T_n, *R_n, *V_n;
  int options_per_thread = numOptions / num_threads;
  S_n = new real[numOptions];
  X_n = new real[numOptions];
  T_n = new real[numOptions];
  R_n = new real[numOptions];
  V_n = new real[numOptions];

  for(int i = 0; i < numOptions; i++)
    {
      int thread_num = i / options_per_thread;
      int item_this_thread = i % options_per_thread;
      S_n[(item_this_thread * num_threads) + thread_num] = S[i];
      X_n[(item_this_thread * num_threads) + thread_num] = X[i];
      R_n[(item_this_thread * num_threads) + thread_num] = R[i];
      V_n[(item_this_thread * num_threads) + thread_num] = V[i];
      T_n[(item_this_thread * num_threads) + thread_num] = T[i];
    }

  free(S);
  free(X);
  free(R);
  free(V);
  free(T);

  S = S_n;
  X = X_n;
  R = R_n;
  V = V_n;
  T = T_n;

}
