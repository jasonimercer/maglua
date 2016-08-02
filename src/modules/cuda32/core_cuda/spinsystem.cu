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

static size_t memTotal()
{
	size_t free, total;
	//cuMemGetInfo(&free, &total);
	cudaMemGetInfo(&free, &total);       

	return total;
}

static size_t memLeft()
{
// 	CUresult res;
	size_t free, total;
	//cuMemGetInfo(&free, &total);
	cudaMemGetInfo(&free, &total); 

	return free;
}

#define CHECKCALL_FL(expression, file, line) \
{ \
	const cudaError_t err = (expression); \
	if(err != cudaSuccess) \
	{ \
		printf("(%s:%i) %s => (%i)%s\n", file, line, #expression, err, cudaGetErrorString(err)); \
		fprintf(logfile,"(%s:%i) %s => (%i)%s\n", file, line, #expression, err, cudaGetErrorString(err)); \
	} \
}

#define CHECKCALL_FLe(lval,expression, file, line)	\
{ \
	lval = (expression); \
	if(lval != cudaSuccess) \
	{ \
		printf("(%s:%i) %s => (%i)%s\n", file, line, #expression, lval, cudaGetErrorString(lval)); \
		fprintf(logfile,"(%s:%i) %s => (%i)%s\n", file, line, #expression, lval, cudaGetErrorString(lval)); \
	} \
}

#define CHECKCALL(expression)  CHECKCALL_FL(expression, __FILE__, __LINE__)
#define CHECKCALLe(lval,expression)  CHECKCALL_FLe(lval,expression, __FILE__, __LINE__)

static FILE* logfile = 0;
cudaError_t malloc_device_(void** d_v, size_t n, const char* file, unsigned int line)
{
    if(!logfile)
    {
	logfile = fopen("malloc.log", "w");
    }

    cudaError_t err;
// 	printf("malloc_device %i bytes\n", n);
    CHECKCALL_FLe(err,cudaMalloc(d_v, n), file, line);

	
	fprintf(logfile, "[%10lu/%10lu] (%s:%i) %8li %p\n", memTotal()-memLeft(), memTotal(), file, line, n, *d_v);
// 	fprintf(logfile, "free %i bytes\n",  memLeft());
	fflush(logfile);

	//TODO here we check for fail and compress if needed
	
	return err; //eventually this will reflect succesfulness of malloc
}

void free_device_(void* d_v, const char* file, unsigned int line)
{
	CHECKCALL_FL(cudaFree(d_v), file,line);
}

cudaError_t malloc_host_(void** h_v, size_t n, const char* file, unsigned int line)
{
    cudaError_t err;
    CHECKCALL_FLe(err,cudaMallocHost(h_v, n),file,line);
    return err; //to mirror malloc_device
}

void free_host_(void* h_v, const char* file, unsigned int line)
{
	CHECKCALL_FL(cudaFreeHost(h_v),file,line);
}





void memcpy_d2d_(void* d_dest, void* d_src, size_t n, const char* file, const unsigned int line)
{
    CHECKCALL_FL(cudaMemcpy(d_dest, d_src, n, cudaMemcpyDeviceToDevice),file,line);
}

void memcpy_d2h_(void* h_dest, void* d_src, size_t n, const char* file, const unsigned int line)
{
    CHECKCALL_FL(cudaMemcpy(h_dest, d_src, n, cudaMemcpyDeviceToHost),file,line);
}

void memcpy_h2d_(void* d_dest, void* h_src, size_t n, const char* file, const unsigned int line)
{
    CHECKCALL_FL(cudaMemcpy(d_dest, h_src, n, cudaMemcpyHostToDevice),file,line);
}


void ss_copyDeviceToHost32_(float* dest, float* src, int nxyz, const char* file, const unsigned int line)
{
	CHECKCALL_FL(cudaMemcpy(dest, src, sizeof(float)*nxyz, cudaMemcpyDeviceToHost), file, line);
}

void ss_copyHostToDevice32_(float* dest, float* src, int nxyz, const char* file, const unsigned int line)
{
// 	printf("%p <- %p (%i)\n", dest, src, nxyz);
	CHECKCALL_FL(cudaMemcpy(dest, src, sizeof(float)*nxyz, cudaMemcpyHostToDevice), file, line);
}




__global__ void scaleValue32(float* dest, int n, float s)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= n)
		return;
	dest[idx] *= s;
}

void ss_d_scale3DArray32(float* d_dest, int n, float s)
{
	const int threads = 256;
	const int blocks = n / threads + 1;

	scaleValue32<<<blocks, threads>>>(d_dest, n, s);
	KCHECK
}

__global__ void scaleaddValue32(float* dest, const int n, const float m1, const float* s1, const float m2, const float* s2)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= n)
		return;
	dest[idx] = m1 * s1[idx] + m2 * s2[idx];
}


void ss_d_add3DArray32(float* d_dest, int nx, int ny, int nz, float* d_src1, float* d_src2)
{
	const int threads = 256;
	const int nxyz = nx*ny*nz;
	const int blocks = nxyz / threads + 1;

	scaleaddValue32<<<blocks, threads>>>(d_dest, nxyz, 1.0, d_src1, 1.0, d_src2);
	KCHECK
}

void cuda_addArrays32(float* d_dest, int n, const float* d_src1, const float* d_src2)
{
	const int threads = 256;
	const int blocks = n / threads + 1;

	scaleaddValue32<<<blocks, threads>>>(d_dest, n, 1.0, d_src1, 1.0, d_src2);
	KCHECK
}

void cuda_scaledAddArrays32(float* d_dest, int n, const float s1, const float* d_src1, const float s2, const float* d_src2)
{
	const int threads = 256;
	const int blocks = n / threads + 1;

	scaleaddValue32<<<blocks, threads>>>(d_dest, n, s1, d_src1, s2, d_src2);
	KCHECK
}


void ss_d_scaleadd3DArray32(float* d_dest, int nxyz, float s1, float* d_src1, float s2, float* d_src2)
{
	const int threads = 256;
	const int blocks = nxyz / threads + 1;

	scaleaddValue32<<<blocks, threads>>>(d_dest, nxyz, s1, d_src1, s2, d_src2);
	KCHECK	
}



__global__ void setArray32(
	float* dest, const int nxyz, float value)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= nxyz)
		return;
	
	dest[idx] = value;
}

void ss_d_set3DArray32_(float* d_v, int nx, int ny, int nz, float value, const char* file, const unsigned int line)
{
	const int threads = 256;
	const int nxyz = nx*ny*nz;
	const int blocks = nxyz / threads + 1;
	
	setArray32<<<blocks, threads>>>(d_v, nxyz, value);
	KCHECK_FL(file, line)
}



__global__ void absDiffArrays32(
	float* dest, float* src1, float* src2, const int nxyz)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= nxyz)
		return;
	
	dest[idx] = fabs(src1[idx] - src2[idx]);
}

void ss_d_absDiffArrays32_(float* d_dest, float* d_src1, float* d_src2, int nxyz, const char* file, const unsigned int line)
{
	const int threads = 256;
	const int blocks = nxyz / threads + 1;
	
	absDiffArrays32<<<blocks, threads>>>(d_dest, d_src1, d_src2, nxyz);
	KCHECK_FL(file, line)
}



template <unsigned int blockSize>
__global__ void reduce_sum_kernel32(float *g_odata, float *g_idata, unsigned int n)
{
	__shared__ float sdata[blockSize];

	// each block of threads will work on blocksize^2 elements
	unsigned int work = blockSize * blockSize;
	
	unsigned int base = blockIdx.x * work;
	unsigned int tid = threadIdx.x;
	float mysum = 0;
// 	sdata[tid] = 0;
	
#pragma unroll
	for(int j=0; j<blockSize; j++)
	{
		const int k = base + j*blockSize + tid;
		if(k < n) mysum += g_idata[k];
	}
	sdata[tid] = mysum;
	__syncthreads();

	//mysum = sdata[tid];
	
	
// #pragma unroll
	for(int j=2; j<=blockSize; j*=2)
	{
// 		const int k = 
		if(tid < blockSize/j)// < n/(j-1))
			sdata[tid] += sdata[tid + blockSize/j];
		__syncthreads();
	}
	
	if(tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}


#define BS 64
float ss_reduce3DArray_sum32(float* d_v, float* d_ws1, float* h_ws1, int nx, int ny, int nz)
{
	const unsigned int n = nx*ny*nz;
	const int work = BS*BS;
	const int blocks = 1 + n / work;
	const int threads = BS;

	reduce_sum_kernel32<BS><<<blocks, threads>>>(d_ws1, d_v, n);
	KCHECK

	
	memcpy_d2h(h_ws1, d_ws1, sizeof(float)*blocks);

	for(int i=1; i<blocks; i++)
		h_ws1[0] += h_ws1[i];
	
	return h_ws1[0];
}



void ss_d_copyArray32(float* d_dest, float* d_src, int nxyz)
{
	cudaMemcpy(d_dest, d_src, sizeof(float)*nxyz, cudaMemcpyDeviceToDevice);
}


