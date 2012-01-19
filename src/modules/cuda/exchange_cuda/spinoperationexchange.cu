#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>
#include <stdio.h>

#include "spinoperationexchange.hpp"

#define KCHECK \
{ \
	const cudaError_t i = cudaGetLastError();\
	if(i) \
		printf("(%s:%i) %s\n",  __FILE__, __LINE__-1, cudaGetErrorString(i));\
}


__global__ void do_exchange(
	const double* d_sx, const double* d_sy, const double* d_sz,
	const double* d_strength, const int* d_neighbour, const int max_neighbours,
	double* d_hx, double* d_hy, double* d_hz,
	const int nxyz
	)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= nxyz) return;
	
	d_hx[i] = 0;
	d_hy[i] = 0;
	d_hz[i] = 0;
	// not all sites have max_neighbours but we've dummied the
	// fromsite and zero'd the strength so it doesn't matter
	for(int j=0; j<max_neighbours; j++)
	{
		const int p = i * max_neighbours + j;
		const int k = d_neighbour[p];
		const double strength = d_strength[p];

		d_hx[i] += strength * d_sx[k];
		d_hy[i] += strength * d_sy[k];
		d_hz[i] += strength * d_sz[k];
	}
}


void cuda_exchange(
	const double* d_sx, const double* d_sy, const double* d_sz,
	const double* d_strength, const int* d_neighbour, const int max_neighbours,
	double* d_hx, double* d_hy, double* d_hz,
	const int nx, const int ny, const int nz
					)
{
	const int nxyz = nx*ny*nz;
	const int threads = 128;
	const int blocks = nxyz / threads + 1;
	
	do_exchange<<<blocks, threads>>>(
			d_sx, d_sy, d_sz,
			d_strength, d_neighbour, max_neighbours,
			d_hx, d_hy, d_hz, 
			nxyz);

	KCHECK;
}



__global__ void do_exchange_compressed(
	const double* d_sx, const double* d_sy, const double* d_sz,
	const ex_compressed_struct* d_LUT, const unsigned char* d_idx, const int max_neighbours,
	double* d_hx, double* d_hy, double* d_hz,
	const int nxyz)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= nxyz) return;
	
	const ex_compressed_struct* e = & d_LUT[ (int)d_idx[i] * max_neighbours ];
	
	d_hx[i] = 0;
	d_hy[i] = 0;
	d_hz[i] = 0;
	for(int j=0; j<max_neighbours; j++)
	{
		const int p = (i + e[j].offset) % nxyz;
		const double strength = e[j].strength;

		d_hx[i] += strength * d_sx[p];
		d_hy[i] += strength * d_sy[p];
		d_hz[i] += strength * d_sz[p];
	}
}


void cuda_exchange_compressed(
	const double* d_sx, const double* d_sy, const double* d_sz,
	const ex_compressed_struct* d_LUT, const unsigned char* d_idx, const int max_neighbours,
	double* d_hx, double* d_hy, double* d_hz, 
	const int nxyz)
{
	const int threads = 256;
	const int blocks = nxyz / threads + 1;

	do_exchange_compressed<<<blocks, threads>>>(
		d_sx, d_sy, d_sz,
		d_LUT, d_idx, max_neighbours,
		d_hx, d_hy, d_hz, 
		nxyz);
	KCHECK;
}
