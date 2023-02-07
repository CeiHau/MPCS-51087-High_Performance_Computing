// #include<iostream>

// #include<math.h>
// #include <random>
// #include <algorithm>
// #include <chrono>
#include <fstream>
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<math.h>
#include<string>
using namespace std;
#define NBLOCKS 125000000
#define NTHREADS_PER_BLOCK 8
#define M_PI           3.14159265358979323846  /* pi */

__global__ void ray_tracing(double *G, int n);
__device__ void direction_sampling(double *V, uint64_t *seed);
__device__ double LCG_random_double(uint64_t *seed);
__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n);

__device__ double dotp( double * a, double * b);
__device__ double square(double a);

__device__ void sub( double *a, double *b, double *c);
__device__ void mult(double *V, double t, double *W);
__device__ double mod(double *V);
__device__ int getGlobalIdx_1D_1D();
__device__ void show(double *V);
void save_to_file(double *C, const string &name, int N);

// /usr/local/cuda/bin/nvcc /home/wxh/Project3/Cuda/cuda_version.cu -o /home/wxh/Project3/Cuda/cuda_version -arch=sm_61
// compile: nvcc cuda_version.cu -o cuda_version -arch=sm_61

int main(int argc, char * argv[]) {
    int n;
    n  = atoi(argv[1]); // I used to set 1000
    // N_rays = atoi(argv[2]);  // I used to set 10000000

    cudaEvent_t                /* CUDA timers */
        start_device,
        stop_device;  
    float time_device;

    /* creates CUDA timers but does not start yet */
    cudaEventCreate(&start_device);
    cudaEventCreate(&stop_device);


    /* device version of vectors */
    double   *dev_G; 

    /* host version of vectors */
    double * G;

    /* allocate host memory */
    G = (double *) malloc(sizeof(double) * n * n);

    // Initialize G, G[i][j] = 0 for all (i, j)
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            G[i * n + j] = 0;
        }
    }
   
    /* allocate device memory */
    cudaMalloc((void **) &dev_G, n * n * sizeof(double));

    /* copy data to device memory */
    cudaMemcpy(dev_G, G, n * n * sizeof(double), cudaMemcpyHostToDevice);

    cudaEventRecord( start_device, 0 );  

    // ray tarcing
    ray_tracing<<< NBLOCKS, NTHREADS_PER_BLOCK>>>(dev_G, n);

    cudaEventRecord( stop_device, 0 );
    cudaEventSynchronize( stop_device );
    cudaEventElapsedTime( &time_device, start_device, stop_device );

    cudaMemcpy(G, dev_G, n * n * sizeof(double), cudaMemcpyDeviceToHost);
    
    printf("time elapsed device: %f(s)\n",  time_device/1000.);
    
    save_to_file(G, "G.csv", n);

    free(G);
    cudaFree(dev_G);

    cudaEventDestroy( start_device );
    cudaEventDestroy( stop_device );
}

__global__ void ray_tracing(double *G, int n) {
    double W[3];
    W[1] = 10;
    double Wmax = 10;
    double L[3] = {4, 4, -1};
    double C[3] = {0, 12, 0};
    double b, R = 6;
    double V[3], I[3], N[3], S[3], I_C[3], L_I[3];
    // Set seed
    uint64_t *seed = (uint64_t *) malloc(sizeof(uint64_t));
    *seed = fast_forward_LCG(0, 200 * getGlobalIdx_1D_1D());

    do {
    
        // Sample random V from unit sphere
        direction_sampling(V, seed);
        // The intersection of the view ray and the window
        W[0] = W[1] / V[1] * V[0];
        W[2] = W[1] / V[1] * V[2];


    } while(!( (abs(W[0]) < Wmax) && (abs(W[2]) < Wmax) && ((square(dotp(V, C)) + square(R) - dotp(C, C)) > 0)) );

    double t  = dotp(V, C) - sqrt(square(dotp(V, C)) + square(R) - dotp(C, C));
    mult(V, t, I); //The intersection of the view ray and the sphere
    sub(I, C, I_C);
    mult(I_C, 1 / mod(I_C), N);

    sub(L, I, L_I);
    mult(L_I, 1/mod(L_I), S);
    b = max(0.0, dotp(S, N));
    // find (i, j) such that G(i, j) is the gridpoint of W
    int i = n - 1 - (W[0] / (2 * Wmax) + 0.5) * n;
    int j = (W[2] / ( 2 * Wmax) + 0.5) * n;
    

    atomicAdd(&G[i * n + j], b);
    free(seed);
}

__device__ void direction_sampling(double *V, uint64_t *seed) {


    // Sample φ from uniform distribution (0, 2π)
    double phi = LCG_random_double(seed);
    phi = phi * M_PI;
    
    // Sample cos(θ) from uniform distribution (−1, 1)
    double cos_theta = LCG_random_double(seed);
    cos_theta = (cos_theta * 2) - 1;
    double sin_theta = sqrt(1 - cos_theta * cos_theta);

    V[0] = sin_theta * cos(phi);
    V[1] = sin_theta * sin(phi);
    V[2] = cos_theta;

    
}

// A 63−bit LCG
// Returns a double precision value from a uniform distribution
// between 0.0 and 1.0 using a caller −owned state variable .
__device__ double LCG_random_double(uint64_t *seed) {
    const uint64_t m = 9223372036854775808ULL; // 2ˆ63
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;
    *seed = (a * (*seed) + c) % m;
    return (double) (*seed) / (double) m;
}

// ”Fast Forwards” an LCG PRNG stream
// seed : starting seed
// n: number of iterations (samples) to forward
// Returns : forwarded seed value
__device__ uint64_t fast_forward_LCG(uint64_t seed, uint64_t n) {
    const uint64_t m = 9223372036854775808ULL; // 2ˆ63
    uint64_t a = 2806196910506780709ULL;
    uint64_t c = 1ULL;
    n = n % m;
    uint64_t a_new = 1;
    uint64_t c_new = 0;
    while (n > 0) {
        if (n & 1) {
            a_new *= a;
            c_new = c_new * a + c;
        }
        c *= (a + 1);
        a *= a;
        n >>= 1;
    }
    return (a_new * seed + c_new) % m;
}

__device__ double dotp( double * a, double * b)
{
    double c = 0.0;
    for( long i = 0; i < 3; i++ )
        c += a[i]*b[i];
    return c;
}


__device__ double square(double a) {
    return a * a;
}

__global__ void daxpy(double *a, double *b, double *c, long n){
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  double alpha = 1.2;
  if (tid < n)
    a[tid] = alpha*b[tid] + c[tid];

  return;
}

__device__ void mult(double *V, double t, double *W) {
    for(int i = 0; i < 3; i++) {
        W[i] = V[i] * t;
    }
}

__device__ void sub( double *a, double *b, double *c) {
    for(int i = 0; i < 3; i++) {
        c[i] = a[i] - b[i];
    }
}
__device__ double mod(double *V) {
    double result = 0;
    for(int i = 0; i < 3; i++) {
        result += (V[i] * V[i]);
    }
    return result;
}

__device__ int getGlobalIdx_1D_1D() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

void save_to_file(double *C, const string &name, int N) {
    ofstream outFile;

    outFile.open(
            "/home/wxh/Project3/parallelize/" +
            name);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            outFile << C[i * N + j] << ", ";

        }
        outFile << "\n";
    }

    outFile << "\n";
    outFile.close();
    std::cout << "generate " + name << std::endl;
}