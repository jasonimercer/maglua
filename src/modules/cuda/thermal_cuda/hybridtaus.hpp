// creating RNG as per
// http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>
#include <stdint.h>
#include <stdlib.h>
typedef unsigned int state_t;

void HybridTausAllocState(state_t** d_state, int nx, int ny, int nz);
void HybridTausAllocRNG(float** d_rngs, int nx, int ny, int nz);

void HybridTausSeed(state_t* d_state, int nx, int ny, int nz, const int i);


void HybridTausFreeState(state_t* d_state);
void HybridTausFreeRNG(float* d_rngs);

void HybridTaus_get6Normals(state_t* d_state, float* d_rngs, const int nx, const int ny, const int nz);
