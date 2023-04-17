// Copyright (c) 2007 Intel Corp.

// Black-Scholes
// Analytical method for calculating European Options
//
// 
// Reference Source: Options, Futures, and Other Derivatives, 3rd Edition, Prentice 
// Hall, John C. Hull,

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <approx.h>
#include <fstream>
// uncomment to output device statistics
//#define APPROX_DEV_STATS 1
//#include <approx_debug.h>
#include <omp.h>
#include <cassert>

#define DOUBLE 0
#define FLOAT 1
#define INT 2
#define LONG 3

//Precision to use for calculations
#define fptype double
#define real fptype

void ArrangeInputsThreadLocality(int numOptions, int num_threads, real *&S, real *&X, real *&T, real *&R, real *&V, int *&otype)
{
  real *S_n, *X_n, *T_n, *R_n, *V_n;
  int options_per_thread = numOptions / num_threads;
  S_n = new real[numOptions];
  X_n = new real[numOptions];
  T_n = new real[numOptions];
  R_n = new real[numOptions];
  V_n = new real[numOptions];
  int *otype_n = new int[numOptions];

  for(int i = 0; i < numOptions; i++)
    {
      int thread_num = i / options_per_thread;
      int item_this_thread = i % options_per_thread;
      S_n[(item_this_thread * num_threads) + thread_num] = S[i];
      X_n[(item_this_thread * num_threads) + thread_num] = X[i];
      R_n[(item_this_thread * num_threads) + thread_num] = R[i];
      V_n[(item_this_thread * num_threads) + thread_num] = V[i];
      T_n[(item_this_thread * num_threads) + thread_num] = T[i];
      otype_n[(item_this_thread * num_threads) + thread_num] = otype[i];
    }

  free(S);
  free(X);
  free(R);
  free(V);
  free(T);
  free(otype);

  S = S_n;
  X = X_n;
  R = R_n;
  V = V_n;
  T = T_n;
  otype = otype_n;

}

void writeQualityFile(char *fileName, void *ptr, int type, size_t numElements){
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



#define DOUBLE 0
#define FLOAT 1
#define INT 2


fptype *prices;
size_t numOptions;

int    * otype;
fptype * sptprice;
fptype * strike;
fptype * rate;
fptype * volatility;
fptype * otime;
int numError = 0;
int nThreads;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Cumulative Normal Distribution Function
// See Hull, Section 11.8, P.243-244
//
//CONSTANTS
#pragma omp declare target
const fptype inv_sqrt_2xPI=0.39894228040143270286f;
const fptype zero = 0.0;
const fptype half = 0.5;
const fptype const1=0.2316419;
const fptype one=1.0;
const fptype const2=0.319381530;
const fptype const3=0.356563782;
const fptype const4=1.781477937;
const fptype const5=1.821255978;
const fptype const6=1.330274429;

fptype CNDF ( fptype InputX ) 
{
    int sign;

    fptype OutputX;
    fptype xInput;
    fptype xNPrimeofX;
    fptype expValues;
    fptype xK2;
    fptype xK2_2, xK2_3;
    fptype xK2_4, xK2_5;
    fptype xLocal, xLocal_1;
    fptype xLocal_2, xLocal_3;
    fptype temp;

    // Check for negative value of InputX
    if (InputX < zero) {
        InputX = -InputX;
        sign = 1;
    } else 
        sign = 0;

    xInput = InputX;

    // Compute NPrimeX term common to both four & six decimal accuracy calcs
    temp = -half * InputX * InputX;

    expValues = exp(temp);

    xNPrimeofX = expValues;
    xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;

    xK2 = const1* xInput;
    xK2 = one + xK2;
    xK2 = one / xK2;
    xK2_2 = xK2 * xK2;
    xK2_3 = xK2_2 * xK2;
    xK2_4 = xK2_3 * xK2;
    xK2_5 = xK2_4 * xK2;

    xLocal_1 = xK2 * const2;
    xLocal_2 = xK2_2 * (-const3);
    xLocal_3 = xK2_3 * const4;
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_4 * (-const5);
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_5 * const6;
    xLocal_2 = xLocal_2 + xLocal_3;

    xLocal_1 = xLocal_2 + xLocal_1;
    xLocal   = xLocal_1 * xNPrimeofX;
    xLocal   = one - xLocal;

    OutputX  = xLocal;

    if (sign) {
        OutputX = one - OutputX;
    }

    return OutputX;
}

fptype BlkSchlsEqEuroNoDiv( fptype sptprice,
        fptype strike, fptype rate, fptype volatility,
        fptype time, float timet )
{
    fptype OptionPrice;

    // local private working variables for the calculation
    fptype xStockPrice;
    fptype xStrikePrice;
    fptype xRiskFreeRate;
    fptype xVolatility;

    fptype xTime;
    fptype xSqrtTime;

    fptype logValues;
    fptype xLogTerm;
    fptype xD1;
    fptype xD2;
    fptype xPowerTerm;
    fptype xDen;
    fptype d1;
    fptype d2;
    fptype FutureValueX;
    fptype NofXd1;
    fptype NofXd2;
    fptype NegNofXd1;
    fptype NegNofXd2;    
    fptype temp;

    xStockPrice = sptprice;
    xStrikePrice = strike;
    xRiskFreeRate = rate;
    xVolatility = volatility;

    xTime = time;
    xSqrtTime = sqrt(xTime);

    temp = sptprice / strike;

    logValues = log( sptprice / strike );

    xLogTerm = logValues;


    xPowerTerm = xVolatility * xVolatility;
    xPowerTerm = xPowerTerm * half;

    xD1 = xRiskFreeRate + xPowerTerm;
    xD1 = xD1 * xTime;
    xD1 = xD1 + xLogTerm;

    xDen = xVolatility * xSqrtTime;
    xD1 = xD1 / xDen;
    xD2 = xD1 -  xDen;

    d1 = xD1;
    d2 = xD2;

    NofXd1 = CNDF( d1 );

    NofXd2 = CNDF( d2 );

    temp = -(rate*time);

    FutureValueX =  ( exp( temp  ) );

    FutureValueX *=strike;

        NegNofXd1 = (one - NofXd1);
        NegNofXd2 = (one - NofXd2);
        OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);

    return OptionPrice;
}
#pragma omp end declare target


int bs_thread(void *tid_ptr) {
    int i, j,k;
    fptype price;
    fptype priceDelta;
    int end = numOptions;

    char *tpb = std::getenv("THREADS_PER_BLOCK");
    char *nblock = std::getenv("NUM_BLOCKS");
    int NTHREADS = atoi(tpb);
    int NUM_TEAMS = atoi(nblock);

#pragma omp target data map(to:sptprice[0:numOptions], strike[0:numOptions], rate[0:numOptions], volatility[0:numOptions], otime[0:numOptions], otype[0:numOptions]) map(tofrom:prices[0:numOptions])
    {

    double tst = omp_get_wtime();
    //@APPROX LABEL("entire") APPROX_TECH(lPerfo|sPerfo)
#pragma omp target teams distribute parallel for num_teams(NUM_TEAMS) num_threads(NTHREADS)
    for (i=0; i<end; i++) {
      //@APPROX LABEL("entire_memo_in") APPROX_TECH(MEMO_IN) IN(sptprice[i], strike[i], rate[i], volatility[i], otime[i], otype[i]) OUT(prices[i])
      //@APPROX LABEL("entire_memo_out") APPROX_TECH(MEMO_OUT) IN(sptprice[i]) OUT(prices[i])
        prices[i] = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
                rate[i], volatility[i], otime[i],
                0);

    }

    double tend = omp_get_wtime();
    printf("Elapsed: %f\n", tend-tst);

    }


      #ifdef APPROX_DEV_STATS
  std::ofstream out_file;
  std::string stat_outfile = "";
  char *stat_outfile_ptr = std::getenv("APPROX_STATS_FILE");
  if(stat_outfile_ptr == nullptr)
    stat_outfile = "thread_statistics.csv";
  else
    stat_outfile = stat_outfile_ptr;
  out_file.open(stat_outfile);
  #ifdef APPROX_IN
  writeDeviceThreadStatistics(out_file);
  #else 
  writeDeviceThreadStatistics(out_file, false);
  #endif
  out_file.close();
  #endif //APPROX_DEV_STATS

    return 0;
}

int main (int argc, char **argv)
{
    FILE *file;
    int i;
    int loopnum;
    int rv;

    // start timer. 


    printf("PARSEC Benchmark Suite\n");
    fflush(NULL);
    if (argc != 4)
    {
        printf("Usage:\n\t%s <nthreads> <inputFile> <outputFile>\n", argv[0]);
        exit(1);
    }
    nThreads = atoi(argv[1]);
    char *inputFile = argv[2];
    char *outputFile = argv[3];

    //Read input data from file
    file = fopen(inputFile, "rb");
    if(file == NULL) {
        printf("ERROR: Unable to open file `%s'.\n", inputFile);
        exit(1);
    }

    if(nThreads != 1) {
        printf("Error: <nthreads> must be 1 (serial version)\n");
        exit(1);
    }

#define PAD 256
#define LINESIZE 64
    readData(file,&otype, &numOptions);  
    readData(file,&sptprice, &numOptions);  
    readData(file,&strike, &numOptions);  
    readData(file,&rate, &numOptions);  
    readData(file,&volatility, &numOptions);  
    readData(file,&otime, &numOptions);  
    prices = (fptype*) malloc(sizeof(fptype)*numOptions);

    char *tpb = std::getenv("THREADS_PER_BLOCK");
    char *nblock = std::getenv("NUM_BLOCKS");
    int NTHREADS = atoi(tpb);
    int NUM_TEAMS = atoi(nblock);

    int num_threads = NTHREADS * NUM_TEAMS;


    #if ADJACENCY_REARRANGE
    ArrangeInputsThreadLocality(numOptions, num_threads, sptprice, strike, otime, rate, volatility, otype);
    #endif // ADJACENCY_REARRANGE


    int tid=0;
    // if we don't do this, then the exact comp. will be unfairly disadvantaged,
    // because OMP will do some startup work on the first transfer, which 
    // approx does during setup.
    #pragma omp target data map(tofrom:prices[0:numOptions])
    {

	#pragma omp target teams distribute parallel for num_threads(256)
      for(int i = 0; i < numOptions; i++)
        prices[i] = 0;
    }

    // startMeasure();
    bs_thread(&tid);


    //Write prices to output file
    writeQualityFile(outputFile, prices, DOUBLE, numOptions);
    delete[] (sptprice);
    delete[] (strike);
    delete[] (rate);
    delete[] (volatility);
    delete[] (otime);
    delete[] (otype);
    free(prices);

    return 0;
}

