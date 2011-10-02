#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>

#include "spinsystem.hpp"
#include <stdio.h>

#define CHECK \
{ \
	if(err) \
		printf("(%s:%i) %s\n",  __FILE__, __LINE__-1, cudaGetErrorString(err));\
}

void ss_d_make3DArray(double** v, int nx, int ny, int nz)
{
	const cudaError_t err = cudaMalloc(v, sizeof(double) * nx * ny * nz);
	CHECK
}

void ss_d_free3DArray(double* v)
{
	const cudaError_t err = cudaFree(v);
	CHECK
}

void ss_h_make3DArray(double** v, int nx, int ny, int nz)
{
	const cudaError_t err = cudaHostAlloc(v, sizeof(double) * nx * ny * nz, 0);
	CHECK
}

void ss_h_free3DArray(double* v)
{
	const cudaError_t err = cudaFreeHost(v);
	CHECK
	if(err)
	{
		long* t = 0;
		*t = 5;
	}
}

void ss_copyDeviceToHost(double* dest, double* src, int nxyz)
{
	const cudaError_t err = cudaMemcpy(dest, src, nxyz, cudaMemcpyDeviceToHost);
	CHECK
}

void ss_copyHostToDevice(double* dest, double* src, int nxyz)
{
	const cudaError_t err = cudaMemcpy(dest, src, nxyz, cudaMemcpyHostToDevice);
	CHECK
}





__global__ void addValue(double* dest, const int nx, const int ny, const int offset, double* s1, double* s2)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(x >= nx || y >= ny)
		return;
	
	const int idx = x + y * nx;
	dest[idx+offset] = s1[idx+offset] + s2[idx+offset];
}
void ss_d_add3DArray(double* d_dest, int nx, int ny, int nz, double* d_src1, double* d_src2)
{
	const int blocksx = nx / 32 + 1;
	const int blocksy = ny / 32 + 1;

	dim3 blocks(blocksx, blocksy);
	dim3 threads(32, 32);

	for(int zz=0; zz<nz; zz++)
	{
		addValue<<<blocks, threads>>>(d_dest, nx, ny, nx*ny*zz, d_src1, d_src2);
	}
}



__global__ void setArray(
	double* dest, const int nx, const int ny, const int nz, double value)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int z = blockDim.z * blockIdx.z + threadIdx.z;
	
	if(x >= nx || y >= ny || z >= nz)
		return;
	
	const int idx = x + y * nx + z * nx*ny;
	dest[idx] = value;
}

void ss_d_set3DArray(double* d_v, int nx, int ny, int nz, double value)
{
	if(nz == 1)
	{
		const int blocksx = nx / 32 + 1;
		const int blocksy = ny / 32 + 1;
	
		dim3 blocks(blocksx, blocksy);
		dim3 threads(32, 32);
		
		setArray<<<blocks, threads>>>(d_v, nx, ny, nz, value);
	}
	else
	{
#warning need to implement 3D
	}
}
