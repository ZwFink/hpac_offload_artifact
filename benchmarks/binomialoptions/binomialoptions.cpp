#include <omp.h>
#include <iostream>


#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <fstream>
#include <cassert>
#ifdef APPROX
#include <approx.h>
#endif
// uncomment to output device statistics
//#define APPROX_DEV_STATS 1
//#include <approx_debug.h>
#include <algorithm>
#include <random>


#include "binomialOptions_common.h"

#define DOUBLE 0
#define FLOAT 1
#define INT 2
#define LONG 3

#define THREADBLOCK_SIZE 256
#define NUM_STEPS 1024
#define ELEMS_PER_THREAD (NUM_STEPS/THREADBLOCK_SIZE)


void writeQualityFile(const char *fileName, void *ptr, int type, size_t numElements){
    FILE *fd = fopen(fileName, "wb");
    assert(fd && "Could Not Open File\n");
    fwrite(&numElements, sizeof(size_t), 1, fd);
    fwrite(&type, sizeof(int), 1, fd);
    if ( type == DOUBLE)
        fwrite(ptr, sizeof(double), numElements, fd);
    else if ( type == FLOAT)
        fwrite(ptr, sizeof(float), numElements, fd);
    else if ( type == INT)
        fwrite(ptr, sizeof(int), numElements, fd);
    else
        assert(0 && "Not supported data type to write\n");
    fclose(fd);
}
void readData(FILE *fd, double **data,  size_t* numElements){
    assert(fd && "File pointer is not valid\n");
    fread(numElements, sizeof(size_t),1,fd);
    size_t elements = *numElements;
    double *ptr = (double*) malloc (sizeof(double)*elements);
    assert(ptr && "Could Not allocate pointer\n");
    *data = ptr;
    size_t i;
    int type;
    fread(&type, sizeof(int), 1, fd); 
    if ( type == DOUBLE){
        fread(ptr, sizeof(double), elements, fd);
    }
    else if ( type == FLOAT){
        float *tmp = (float*) malloc (sizeof(float)*elements);
        fread(tmp, sizeof(float), elements,fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (double) tmp[i];
        }
        free (tmp);
    }
    else if( type == INT ){
        int *tmp = (int*) malloc (sizeof(int)*elements);
        fread(tmp, sizeof(int), elements, fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (double) tmp[i];
        }
        free(tmp);
    }
    return; 
}

void readData(FILE *fd, float **data,  size_t* numElements){
    assert(fd && "File pointer is not valid\n");
    fread(numElements, sizeof(size_t),1,fd);
    size_t elements = *numElements;

    float *ptr = (float*) malloc (sizeof(float)*elements);
    assert(ptr && "Could Not allocate pointer\n");
    *data = ptr;

    size_t i;
    int type;
    fread(&type, sizeof(int), 1, fd); 
    if ( type == FLOAT ){
        fread(ptr, sizeof(float), elements, fd);
    }
    else if ( type == DOUBLE){
        double *tmp = (double*) malloc (sizeof(double)*elements);
        fread(tmp, sizeof(double), elements,fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (float) tmp[i];
        }
        free (tmp);
    }
    else if ( type == INT ){
        int *tmp = (int*) malloc (sizeof(int) * elements);
        fread(tmp, sizeof(int), elements, fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (float) tmp[i];
        }
        free(tmp);
    }
    return; 
}

void readData(FILE *fd, int **data,   size_t* numElements){
    assert(fd && "File pointer is not valid\n");
    fread(numElements, sizeof(size_t),1,fd);
    size_t elements = *numElements;

    int *ptr = (int*) malloc (sizeof(int)*elements);
    assert(ptr && "Could Not allocate pointer\n");
    *data = ptr;

    size_t i;
    int type;
    fread(&type, sizeof(int), 1, fd); 
    if ( type == INT ){
        fread(ptr, sizeof(int), elements, fd);
    }
    else if ( type == DOUBLE){
        double *tmp = (double*) malloc (sizeof(double)*elements);
        fread(tmp, sizeof(double), elements,fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (int) tmp[i];
        }
        free (tmp);
    }
    else if( type == FLOAT ){
        float *tmp = (float*) malloc (sizeof(float)*elements);
        fread(tmp, sizeof(float), elements, fd);
        for ( i = 0; i < elements; i++){
            ptr[i] = (int) tmp[i];
        }
        free(tmp);
    }
    return; 
}

////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////

real randData(real low, real high)
{
    real t = (real)rand() / (real)RAND_MAX;
    return ((real)1.0 - t) * low + t * high;
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
  FILE *file;

  if(!(argc == 4 || argc == 5))
    {
      std::cout << "USAGE: " << argv[0] << " input_file num_steps seed [output_file]";
      return EXIT_FAILURE;
    }

  char *inputFile = argv[1];
  int seed = std::atoi(argv[3]);

    //Read input data from file
    file = fopen(inputFile, "rb");
    if(file == NULL) {
        printf("ERROR: Unable to open file `%s'.\n", inputFile);
        exit(1);
    }

  bool write_output = false;
  std::string ofname;
  if(argc == 5)
    {
      write_output = true;
      ofname = argv[4];
    }


  // sptprice
  real *S;
  // strike
  real *X;
  // time
  real *T;
  // rate
  real *R;
  // volatility
  real *V;
  int *otype;

  real
    sumDelta, sumRef, gpuTime, errorVal;

  printf("Reading input data...\n");
  size_t numOptions = 0;

#define PAD 256
#define LINESIZE 64
    readData(file,&otype, &numOptions);  
    readData(file,&S, &numOptions);  
    readData(file,&X, &numOptions);  
    readData(file,&R, &numOptions);  
    readData(file,&V, &numOptions);  
    readData(file,&T, &numOptions);  


    delete[] otype;
    std::fill(V, V+numOptions, 0.10f);


  const int NUM_OPTIONS = numOptions;

  real *callValueBS = new real[NUM_OPTIONS];
  real *callValue = new real[NUM_OPTIONS];

  printf("[%s] - Starting with %d options and %d iterations...\n", argv[0], NUM_OPTIONS, NUM_STEPS);

  const int THREADS_PER_BLOCK = atoi(std::getenv("THREADS_PER_BLOCK"));
  const int NUM_BLOCKS = atoi(std::getenv("NUM_BLOCKS"));
  int num_threads = NUM_BLOCKS*THREADS_PER_BLOCK;

  #if ADJACENCY_REARRANGE
  ArrangeInputsThreadLocality(numOptions, num_threads, S, X, T, R, V);
  #endif // ADJACENCY_REARRANGE

  for (int i = 0; i < NUM_OPTIONS; i++)
    {
      BlackScholesCall(callValueBS[i], S[i], X[i], T[i], R[i], V[i]);
    }

  // warmup kernel -- initialize OMP offload state
  warmup();

  printf("Running offload binomial tree...\n");

  real *vDt_all = new real[numOptions];
  real *puByDf_all = new real[numOptions];
  real *pdByDf_all = new real[numOptions];

  double elapsed = 0.0;
  double tend = 0.0;
  double tst = omp_get_wtime();
  // preprocess for the GPU

  for(int i = 0; i < numOptions; i++)
    {
        const real      _T = T[i];
        const real      _R = R[i];
        const real      _V = V[i];

        const real     dt = _T / (real)NUM_STEPS;
        const real    vDt = _V * sqrt(dt);
        const real    rDt = _R * dt;
        //Per-step interest and discount factors
        const real     If = exp(rDt);
        const real     Df = exp(-rDt);
        //Values and pseudoprobabilities of upward and downward moves
        const real      u = exp(vDt);
        const real      d = exp(-vDt);
        const real     pu = (If - d) / (u - d);
        const real     pd = (real)1.0 - pu;
        const real puByDf = pu * Df;
        const real pdByDf = pd * Df;

        vDt_all[i] = (real) vDt;
        puByDf_all[i] = (real) puByDf;
        pdByDf_all[i] = (real) pdByDf;
    }
  tend = omp_get_wtime();
  elapsed = tend-tst;

#pragma omp target data map(to:S[0:NUM_OPTIONS], X[0:NUM_OPTIONS], vDt_all[0:NUM_OPTIONS], puByDf_all[0:NUM_OPTIONS], pdByDf_all[0:NUM_OPTIONS]) map(from:callValue[0:NUM_OPTIONS])
    {
      tst = omp_get_wtime();
#pragma omp target teams num_teams(NUM_BLOCKS)
#pragma omp parallel num_threads(THREADS_PER_BLOCK)
        {
          for(int k = omp_get_team_num(); k < NUM_OPTIONS; k+=omp_get_num_teams())
          {
          real _callValue = 0;
          real call_exchange[THREADBLOCK_SIZE + 1];
          const int     tid = omp_get_thread_num();
          #pragma omp allocate(call_exchange) allocator(omp_pteam_mem_alloc)
      //@APPROX LABEL("entire_memo_out") APPROX_TECH(MEMO_OUT) IN(_callValue) OUT(_callValue)
	  {

          const real      _S = S[k];
          const real      _X = X[k];
          const real    vDt = vDt_all[k];
          const real puByDf = puByDf_all[k];
          const real pdByDf = pdByDf_all[k];

          real call[ELEMS_PER_THREAD + 1];

      //@APPROX LABEL("entire_memo_in") APPROX_TECH(MEMO_IN) IN(_S, _X, vDt, puByDf, pdByDf) OUT(_callValue)
          {

          #pragma unroll
          for(int i = 0; i < ELEMS_PER_THREAD; ++i)
            call[i] = expiryCallValue(NUM_STEPS, _S, _X, vDt, tid * ELEMS_PER_THREAD + i);

          if (tid == 0)
            call_exchange[THREADBLOCK_SIZE] = expiryCallValue(NUM_STEPS, _S, _X, vDt, NUM_STEPS);

          int final_it = std::max(0, tid * ELEMS_PER_THREAD - 1);

          #pragma unroll 16
          for(int i = NUM_STEPS; i > 0; --i)
            {
              call_exchange[tid] = call[0];
              #pragma omp barrier
              call[ELEMS_PER_THREAD] = call_exchange[tid + 1];
              #pragma omp barrier

              if (i > final_it)
                {
                  #pragma unroll
                  for(int j = 0; j < ELEMS_PER_THREAD; ++j)
                    call[j] = puByDf * call[j + 1] + pdByDf * call[j];
                }
            }
          _callValue = call[0];
          }

	}
          if (tid == 0)
            {
              callValue[k] = _callValue;
            }
        }
	}

        tend = omp_get_wtime();
        elapsed += tend-tst;
    }

    double cpuTime = elapsed;

    printf("binomialOptionsOffload() time: %f sec\n", cpuTime);
    printf("Options per second       : %f     \n", NUM_OPTIONS / (cpuTime));


    printf("Comparing the results...\n");

    printf("CPU binomial vs. Black-Scholes\n");
    sumDelta = 0;
    sumRef   = 0;

    for (int i = 0; i < NUM_OPTIONS; i++)
    {
      sumDelta += fabs(callValueBS[i]- callValue[i]);
        sumRef += fabs(callValueBS[i]);
    }

    if (sumRef >1E-5)
    {
        printf("L1 norm: %E\n", sumDelta / sumRef);
    }
    else
    {
        printf("Avg. diff: %E\n", (double)(sumDelta / (real)NUM_OPTIONS));
    }

    delete[] S;
    delete[] X;
    delete[] T;
    delete[] R;
    delete[] V;

    if(write_output)
      {
        writeQualityFile(ofname.c_str(), callValue, DOUBLE, NUM_OPTIONS);
      }

  #if defined(APPROX) && defined(APPROX_DEV_STATS)
  std::ofstream out_file;
  std::string stat_outfile = "";
  char *stat_outfile_ptr = std::getenv("APPROX_STATS_FILE");
  if(stat_outfile_ptr == nullptr)
    stat_outfile = "thread_statistics.csv";
  else
    stat_outfile = stat_outfile_ptr;
  out_file.open(stat_outfile);
  writeDeviceThreadStatistics(out_file);
  out_file.close();
  #endif //APPROX_DEV_STATS

    delete[] callValue;

    exit(EXIT_SUCCESS);
}
