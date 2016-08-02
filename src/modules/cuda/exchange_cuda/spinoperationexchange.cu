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

#if 0
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

#endif



// multiple spinsystem versions
__global__ void do_exchange_N(
	const double** d_sx_N, const double** d_sy_N, const double** d_sz_N, const double** d_sm_N,
	const double* d_strength, const int* d_neighbour, const int max_neighbours,
	double** d_hx_N, double** d_hy_N, double** d_hz_N, const double global_scale,
	const int nxyz, const int n
	)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= nxyz) return;
	
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if(j >= n) return;
	
	const double* d_sx = d_sx_N[j];
	const double* d_sy = d_sy_N[j];
	const double* d_sz = d_sz_N[j];
	const double* d_sm = d_sm_N[j];

	double* d_hx = d_hx_N[j];
	double* d_hy = d_hy_N[j];
	double* d_hz = d_hz_N[j];
	
	d_hx[i] = 0;
	d_hy[i] = 0;
	d_hz[i] = 0;
	// not all sites have max_neighbours but we've dummied the
	// fromsite and zero'd the strength so it doesn't matter
	for(int k=0; k<max_neighbours; k++)
	{
		const int p = i * max_neighbours + k;
		const int k = d_neighbour[p];
		if(d_sm[k] > 0)
		{
			const double strength = d_strength[p] * global_scale / d_sm[k];

			d_hx[i] += strength * d_sx[k];
			d_hy[i] += strength * d_sy[k];
			d_hz[i] += strength * d_sz[k];
		}
	}
}

void cuda_exchange_N(
	const double** d_sx, const double** d_sy, const double** d_sz, const double** d_sm,
	const double* d_strength, const int* d_neighbour, const int max_neighbours,
	double** d_hx, double** d_hy, double** d_hz, const double global_scale,
	const int nx, const int ny, const int nz,
	const int n)
{
	const int nxyz = nx*ny*nz;
	const int threadsX = 128;
	const int blocksX = nxyz / threadsX + 1;

	const int threadsY = 1;
	const int blocksY = n;
	
	dim3 gd(blocksX, blocksY);
	dim3 bd(threadsX, threadsY);

	do_exchange_N<<<gd, bd>>>(
			d_sx, d_sy, d_sz, d_sm,
			d_strength, d_neighbour, max_neighbours,
			d_hx, d_hy, d_hz, global_scale,
			nxyz, n);

	KCHECK;
}






__global__ void do_exchange_compressed_N(
	const double** d_sx_N, const double** d_sy_N, const double** d_sz_N, const double** d_sm_N,
	const ex_compressed_struct* d_LUT, const unsigned char* d_idx, const int max_neighbours,
	double** d_hx_N, double** d_hy_N, double** d_hz_N, const double global_scale,
	const int nxyz, const int n)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= nxyz) return;
	
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if(j >= n) return;
	
	const double* d_sx = d_sx_N[j];
	const double* d_sy = d_sy_N[j];
	const double* d_sz = d_sz_N[j];
	const double* d_sm = d_sm_N[j];

	double* d_hx = d_hx_N[j];
	double* d_hy = d_hy_N[j];
	double* d_hz = d_hz_N[j];
	
	const ex_compressed_struct* e = & d_LUT[ (int)d_idx[i] * max_neighbours ];
	
	d_hx[i] = 0;
	d_hy[i] = 0;
	d_hz[i] = 0;
	for(int k=0; k<max_neighbours; k++)
	{
		const int p = (i + e[k].offset) % nxyz;
		if(d_sm[p] > 0)
		{
			const double strength = e[k].strength * global_scale  / d_sm[p];

			d_hx[i] += strength * d_sx[p];
			d_hy[i] += strength * d_sy[p];
			d_hz[i] += strength * d_sz[p];
		}
	}
}


void cuda_exchange_compressed_N(
	const double** d_sx, const double** d_sy, const double** d_sz, const double** d_sm,
	const ex_compressed_struct* d_LUT, const unsigned char* d_idx, const int max_neighbours,
	double** d_hx, double** d_hy, double** d_hz, const double global_scale,
	const int nxyz, const int n)
{
	const int threadsX = 256;
	const int blocksX = nxyz / threadsX + 1;

	const int threadsY = 1;
	const int blocksY = n;
	
	dim3 gd(blocksX, blocksY);
	dim3 bd(threadsX, threadsY);

	do_exchange_compressed_N<<<gd, bd>>>(
		d_sx, d_sy, d_sz, d_sm,
		d_LUT, d_idx, max_neighbours,
		d_hx, d_hy, d_hz, global_scale,
		nxyz, n);
	KCHECK;
}



