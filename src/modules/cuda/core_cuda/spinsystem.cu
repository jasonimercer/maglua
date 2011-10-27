#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>

#include "spinsystem.hpp"
#include <stdio.h>

#define KCHECK \
{ \
	const cudaError_t i = cudaGetLastError();\
	if(i) \
		printf("(%s:%i) %s\n",  __FILE__, __LINE__-1, cudaGetErrorString(i));\
}

#define KCHECK_FL(f,l) \
{ \
	const cudaError_t i = cudaGetLastError();\
	if(i) \
		printf("(%s:%i) %s\n",  f, l, cudaGetErrorString(i));\
}


#define SEGFAULT \
{ \
	long* i = 0; \
	*i = 5; \
}

#define CHECKCALL_FL(expression, file, line) \
{ \
	const cudaError_t err = (expression); \
	if(err != cudaSuccess) \
	{ \
		printf("(%s:%i) %s => (%i)%s\n", file, line, #expression, err, cudaGetErrorString(err)); \
	} \
}

#define CHECKCALL(expression)  CHECKCALL_FL(expression, __FILE__, __LINE__)


void ss_d_make3DArray(double** v, int nx, int ny, int nz)
{
	CHECKCALL(cudaMalloc(v, sizeof(double) * nx * ny * nz));
}

void ss_d_free3DArray(double* v)
{
	CHECKCALL(cudaFree(v));
}

void ss_h_make3DArray(double** v, int nx, int ny, int nz)
{
	CHECKCALL(cudaMallocHost(v, sizeof(double) * nx * ny * nz));
}

void ss_h_free3DArray(double* v)
{
	CHECKCALL(cudaFreeHost(v));
}

void ss_copyDeviceToHost_(double* dest, double* src, int nxyz, const char* file, const unsigned int line)
{
	CHECKCALL_FL(cudaMemcpy(dest, src, sizeof(double)*nxyz, cudaMemcpyDeviceToHost), file, line);
}

void ss_copyHostToDevice_(double* dest, double* src, int nxyz, const char* file, const unsigned int line)
{
// 	printf("%p <- %p (%i)\n", dest, src, nxyz);
	CHECKCALL_FL(cudaMemcpy(dest, src, sizeof(double)*nxyz, cudaMemcpyHostToDevice), file, line);
}





__global__ void addValue(double* dest, const int n, double* s1, double* s2)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= n)
		return;
	dest[idx] = s1[idx] + s2[idx];
}

void ss_d_add3DArray(double* d_dest, int nx, int ny, int nz, double* d_src1, double* d_src2)
{
	const int threads = 256;
	const int nxyz = nx*ny*nz;
	const int blocks = nxyz / threads + 1;

	addValue<<<blocks, threads>>>(d_dest, nxyz, d_src1, d_src2);
	KCHECK
}


__global__ void setArray(
	double* dest, const int nxyz, double value)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= nxyz)
		return;
	
	dest[idx] = value;
}

void ss_d_set3DArray_(double* d_v, int nx, int ny, int nz, double value, const char* file, const unsigned int line)
{
	const int threads = 256;
	const int nxyz = nx*ny*nz;
	const int blocks = nxyz / threads + 1;
	
	setArray<<<blocks, threads>>>(d_v, nxyz, value);
	KCHECK_FL(file, line)
}



__global__ void absDiffArrays(
	double* dest, double* src1, double* src2, const int nxyz)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= nxyz)
		return;
	
	dest[idx] = fabs(src1[idx] - src2[idx]);
}

void ss_d_absDiffArrays_(double* d_dest, double* d_src1, double* d_src2, int nxyz, const char* file, const unsigned int line)
{
	const int threads = 256;
	const int blocks = nxyz / threads + 1;
	
	absDiffArrays<<<blocks, threads>>>(d_dest, d_src1, d_src2, nxyz);
	KCHECK_FL(file, line)
}



template <unsigned int blockSize>
__device__ void warpReduce(volatile double *sdata, unsigned int tid)
{
	if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
	if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
	if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
	if (blockSize >=   8) sdata[tid] += sdata[tid +  4];
	if (blockSize >=   4) sdata[tid] += sdata[tid +  2];
	if (blockSize >=   2) sdata[tid] += sdata[tid +  1];
}

template <unsigned int blockSize>
__global__ void reduce_sum_kernel(double *g_odata, double *g_idata, unsigned int n)
{
// 	extern __shared__ double sdata[];
	__shared__ double sdata[blockSize];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;
	while (i < n)
	{
		sdata[tid] += g_idata[i] + g_idata[i+blockSize];
		i += gridSize;
	}

	__syncthreads();
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] += sdata[tid + 256];
		}
	__syncthreads();
	}
	
	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] += sdata[tid + 128];
		}
		__syncthreads();
	}
	if (blockSize >= 128)
	{
		if (tid <   64)
		{
			sdata[tid] += sdata[tid +   64];
		}
		__syncthreads();
	}

	if(tid < 32)
		warpReduce<blockSize>(sdata, tid);
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

#define BS 64
double ss_reduce3DArray_sum(double* d_v, double* d_ws1, double* h_ws1, int nx, int ny, int nz)
{
	int blocks = (nx*ny*nz) / BS + 1;
	
	unsigned int n = nx*ny*nz;
	
	reduce_sum_kernel<BS><<<blocks, BS>>>(d_ws1, d_v, n);
	KCHECK

	CHECKCALL(cudaMemcpy(h_ws1, d_ws1, sizeof(double)*blocks, cudaMemcpyDeviceToHost));
	
	
	for(int i=1; i<blocks; i++)
		h_ws1[0] += h_ws1[i];
	return h_ws1[0];
}



void ss_d_copyArray(double* d_dest, double* d_src, int nxyz)
{
	cudaMemcpy(d_dest, d_src, sizeof(double)*nxyz, cudaMemcpyDeviceToDevice);
}


