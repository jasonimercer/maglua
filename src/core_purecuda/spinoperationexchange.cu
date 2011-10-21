#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>
#include <stdio.h>

#include "spinoperationexchange.hpp"



#define IDX_PATT(a, b) \
	const int a = blockDim.x * blockIdx.x + threadIdx.x; \
	const int b = blockDim.y * blockIdx.y + threadIdx.y;


#define CHECK \
{ \
	const cudaError_t i = cudaGetLastError();\
	if(i) \
		printf("(%s:%i) %s\n",  __FILE__, __LINE__-1, cudaGetErrorString(i));\
}





void ex_d_makeStrengthArray(double** d_v, int nx, int ny, int nz, int max_neighbours)
{
	cudaMalloc(d_v, sizeof(double) * max_neighbours * nx * ny * nz);
	CHECK
}

void ex_d_freeStrengthArray(double* d_v)
{
	cudaFree(d_v);
	CHECK
}

void ex_d_makeNeighbourArray(int** d_v, int nx, int ny, int nz, int max_neighbours)
{
	cudaMalloc(d_v, sizeof(int) * max_neighbours * nx * ny * nz);
	CHECK
}

void ex_d_freeNeighbourArray(int* d_v)
{
	cudaFree(d_v);
	CHECK
}

void ex_h_makeStrengthArray(double** h_v, int nx, int ny, int nz, int max_neighbours)
{
	cudaHostAlloc(h_v, sizeof(double) * max_neighbours * nx * ny * nz, 0);
	CHECK
}

void ex_h_freeStrengthArray(double* h_v)
{
	cudaFreeHost(h_v);
	CHECK
}

void ex_h_makeNeighbourArray(int** h_v, int nx, int ny, int nz, int max_neighbours)
{
	cudaHostAlloc(h_v, sizeof(int) * max_neighbours * nx * ny * nz, 0);
	CHECK
}

void ex_h_freeNeighbourArray(int* h_v)
{
	cudaFreeHost(h_v);
	CHECK
}

void ex_hd_syncStrengthArray(double* d_v, double* h_v, int nx, int ny, int nz, int max_neighbours)
{
	cudaMemcpy(d_v, h_v, sizeof(double)* max_neighbours * nx * ny * nz, cudaMemcpyHostToDevice);
	CHECK
}

void ex_hd_syncNeighbourArray(int* d_v, int* h_v, int nx, int ny, int nz, int max_neighbours)
{
	cudaMemcpy(d_v, h_v, sizeof(int) * max_neighbours * nx * ny * nz, cudaMemcpyHostToDevice);
	CHECK
}

#define HARD_CODE(n) \
for(int j=0; j<n; j++) \
{ \
	const int p = i * max_neighbours + j; \
	const int k = d_neighbour[p]; \
	const double strength = d_strength[p];	\
			 \
	d_hx[i] += strength * d_sx[k]; \
	d_hy[i] += strength * d_sy[k]; \
	d_hz[i] += strength * d_sz[k]; \
}

__global__ void do_exchange(
	const double* d_sx, const double* d_sy, const double* d_sz,
	const double* d_strength, const int* d_neighbour, const int max_neighbours,
	double* d_hx, double* d_hy, double* d_hz,
	const int nx, const int ny, const int offset
	)
{
	IDX_PATT(x,y)
	
	if(x >= nx || y >= ny)
		return;
	
	const int i = x + y*nx + offset*nx*ny;
	
	switch(max_neighbours)
	{
		case 1:
		#pragma unroll
		HARD_CODE(1)
		break;

		case 2:
		#pragma unroll
		HARD_CODE(2)
		break;

		case 3:
		#pragma unroll
		HARD_CODE(3)
		break;

		case 4:
		#pragma unroll
		HARD_CODE(4)
		break;
		
		case 5:
		#pragma unroll
		HARD_CODE(5)
		break;
				
		case 6:
		#pragma unroll
		HARD_CODE(6)
		break;

		case 8:
		#pragma unroll
		HARD_CODE(8)
		break;

		case 10:
		#pragma unroll
		HARD_CODE(10)
		break;

		case 12:
		#pragma unroll
		HARD_CODE(12)
		break;

		default:
			for(int j=0; j<max_neighbours; j++)
			{
				const int p = i * max_neighbours + j;
				const int k = d_neighbour[p];
				const double strength = d_strength[p];
				if(k >= 0)
				{
					d_hx[i] += strength * d_sx[k];
					d_hy[i] += strength * d_sy[k];
					d_hz[i] += strength * d_sz[k];
				}
			}
	}
	
}

void cuda_exchange(
	const double* d_sx, const double* d_sy, const double* d_sz,
	const double* d_strength, const int* d_neighbour, const int max_neighbours,
	double* d_hx, double* d_hy, double* d_hz,
	const int nx, const int ny, const int nz
					)
{
	const int blocksx = nx / 32 + 1;
	const int blocksy = ny / 32 + 1;

	dim3 blocks(blocksx, blocksy);
	dim3 threads(32, 32);

	for(int z=0; z<nz; z++)
	{
		const int offset = z * nx * nz;
		do_exchange<<<blocks, threads>>>(
				d_sx, d_sy, d_sz,
				d_strength, d_neighbour, max_neighbours,
				d_hx, d_hy, d_hz, 
				nx, ny, offset);
	}
}


