#include "hybridtaus.hpp"

#include <stdio.h>
#include <stdlib.h>
static int (*globalRand)() = rand;

#define CHECK \
{ \
	if(err) \
		printf("(%s:%i) %s\n",  __FILE__, __LINE__-1, cudaGetErrorString(err));\
}

#define IDX_PATT(a) \
	const int a = blockDim.x * blockIdx.x + threadIdx.x;

	
__device__ inline void TausStep(state_t &z, const int S1, const int S2, const int S3, const state_t M)  
{  
	state_t b=(((z << S1) ^ z) >> S2);  
	z = (((z & M) << S3) ^ b);  
// 	return z;
} 

__device__ inline void LCGStep(state_t &z, const state_t A, const state_t C)  
{  
	z = (A*z+C);
// 	return z;
}

// uniform [0:1]
// I: i is offset into rngs, it can be 0 through 5
template <int i>
__global__ void HybridTausStep_I(const int n, state_t* d_state, float* d_rngs)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	if(x >= n)
		return;

	state_t& z1 = d_state[ 0*n + x ];
	state_t& z2 = d_state[ 1*n + x ];
	state_t& z3 = d_state[ 2*n + x ];
	state_t& z4 = d_state[ 3*n + x ];

	TausStep(z1, 13, 19, 12, 4294967294ULL);
	TausStep(z2,  2, 25,  4, 4294967288ULL);
	TausStep(z3,  3, 11, 17, 4294967280ULL);
	 LCGStep(z4, 1664525, 1013904223UL);
	
	const unsigned long V = (z1 ^ z2 ^ z3) + 1;
// 	long V = (z1 ^ z2 ^ z3 ^ z4) | 1;
// 	if(!V) V = 1;
// // 	__fdividef(__uint2float_ru((UINT32)(*x)),(FLOAT)0x100000000);
// // 		          2.3283064365387e-10 *              // Periods  
// 	const state_t V =
// 			((TausStep(z1, 13, 19, 12, 4294967294ULL) //^  // p1=2^31-1  
// // 			  TausStep(z2,  2, 25,  4, 4294967288ULL) ^  // p2=2^30-1  
// // 			  TausStep(z3,  3, 11, 17, 4294967280ULL)   // p3=2^28-1  
// // 			   LCGStep(z4, 1664525, 1013904223UL)      // p4=2^32 
// 			)& 0xFFFFFFFF);
			
// 	d_rngs[idx6 + i] =  __fdividef(__ull2float_rz(V), (float)0x100000001);
	d_rngs[i*n+x] =  __fdividef(__ull2float_rz(V), (float)0x100000001);
// 	d_rngs[idx6 + i] =  __fdividef(__uint2float_rz(V), (float)0x100000001);
}

// transforms uniform numbers into normal numbers
template <int i2>
__global__ void HybridTaus_BoxMuller(const int n, float* d_rngs)
{
// 	return;
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	if(x >= n)
		return;
		
	const int a = (2*i2+0) * n + x;
	const int b = (2*i2+1) * n + x;
	
	const float u0=d_rngs[a];
	const float u1=d_rngs[b];
	const float r=sqrt(-2 * log(u0));  
 
	const float theta=2.0*3.14159265358979*u1;  

	float sint, cost;
	
	__sincosf(theta, &sint, &cost);
	
	d_rngs[a] = r * sint;
	d_rngs[b] = r * cost;
}

void HybridTausAllocState(state_t** d_state, int nx, int ny, int nz)
{
	const cudaError_t err = cudaMalloc(d_state, sizeof(state_t) * 4 * nx * ny * nz);
	CHECK
}

void HybridTausAllocRNG(float** d_rngs, int nx, int ny, int nz)
{
	const cudaError_t err = cudaMalloc(d_rngs, sizeof(float) * 6 * nx * ny * nz);
	CHECK
}

void HybridTausFreeState(state_t* d_state)
{
	const cudaError_t err = cudaFree(d_state);
	CHECK
}

void HybridTausFreeRNG(float* d_rngs)
{
	const cudaError_t err = cudaFree(d_rngs);
	CHECK
}


void HybridTausSeed(state_t* d_state, int nx, int ny, int nz, const int i)
{
	state_t* h_state;

	const int sz = sizeof(state_t) * 4 * nx * ny * nz;
	cudaError_t err = cudaHostAlloc(&h_state, sz, 0);
	CHECK

	#ifndef _WIN32
	char randstate[4096];
	random_data buf;
	initstate_r(i, randstate, 256, &buf);
	#else
	srand( i );
	#endif

	for(int i=0; i<4*nx*ny*nz; i++)
	{
		int32_t a;
		do
		{
			#ifndef _WIN32
			random_r(&buf, &a);
			#else
			a = 0xFFFFFFFF & (globalRand() ^ (globalRand() << 16));
			#endif
		}while(a < 128); // restriction on tausworth init state: > 128

		h_state[i] = a;
	}
	
	err = cudaMemcpy(d_state, h_state, sz, cudaMemcpyHostToDevice);
	CHECK

	err = cudaFreeHost(h_state);
	CHECK
}

#undef CHECK
#define CHECK \
{ \
	const cudaError_t i = cudaGetLastError();\
	if(i) \
		printf("(%s:%i) %s\n",  __FILE__, __LINE__-1, cudaGetErrorString(i));\
}

void HybridTaus_get6Normals(state_t* d_state, float* d_rngs, const int nx, const int ny, const int nz)
{
	const int nxyz = nx * ny * nz;
	const int threads = 512;
	const int blocks = nxyz / threads + 1;
	

	HybridTausStep_I<0><<<blocks, threads>>>(nxyz, d_state, d_rngs);
	CHECK
	HybridTausStep_I<1><<<blocks, threads>>>(nxyz, d_state, d_rngs);
	CHECK
	HybridTausStep_I<2><<<blocks, threads>>>(nxyz, d_state, d_rngs);
	CHECK
	HybridTausStep_I<3><<<blocks, threads>>>(nxyz, d_state, d_rngs);
	CHECK
	HybridTausStep_I<4><<<blocks, threads>>>(nxyz, d_state, d_rngs);
	CHECK
	HybridTausStep_I<5><<<blocks, threads>>>(nxyz, d_state, d_rngs);
	CHECK

			
	HybridTaus_BoxMuller<0><<<blocks, threads>>>(nxyz, d_rngs);
	CHECK
	HybridTaus_BoxMuller<1><<<blocks, threads>>>(nxyz, d_rngs);
	CHECK
	HybridTaus_BoxMuller<2><<<blocks, threads>>>(nxyz, d_rngs);
	CHECK

	
}
