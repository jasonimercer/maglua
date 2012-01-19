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
// 	CUresult res;
	size_t free, total;
	cuMemGetInfo(&free, &total);

	return total;
}

static size_t memLeft()
{
// 	CUresult res;
	size_t free, total;
	cuMemGetInfo(&free, &total);

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


void ss_copyDeviceToHost_(double* dest, double* src, int nxyz, const char* file, const unsigned int line)
{
	CHECKCALL_FL(cudaMemcpy(dest, src, sizeof(double)*nxyz, cudaMemcpyDeviceToHost), file, line);
}

void ss_copyHostToDevice_(double* dest, double* src, int nxyz, const char* file, const unsigned int line)
{
// 	printf("%p <- %p (%i)\n", dest, src, nxyz);
	CHECKCALL_FL(cudaMemcpy(dest, src, sizeof(double)*nxyz, cudaMemcpyHostToDevice), file, line);
}




__global__ void scaleValue(double* dest, int n, double s)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= n)
		return;
	dest[idx] *= s;
}

void ss_d_scale3DArray(double* d_dest, int n, double s)
{
	const int threads = 256;
	const int blocks = n / threads + 1;

	scaleValue<<<blocks, threads>>>(d_dest, n, s);
	KCHECK
}

// __global__ void addValue(double* dest, const int n, double* s1, double* s2)
// {
// 	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
// 	
// 	if(idx >= n)
// 		return;
// 	dest[idx] = s1[idx] + s2[idx];
// }

__global__ void scaleaddValue(double* dest, const int n, const double m1, const double* s1, const double m2, const double* s2)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= n)
		return;
	dest[idx] = m1 * s1[idx] + m2 * s2[idx];
}


void ss_d_add3DArray(double* d_dest, int nx, int ny, int nz, double* d_src1, double* d_src2)
{
	const int threads = 256;
	const int nxyz = nx*ny*nz;
	const int blocks = nxyz / threads + 1;

	scaleaddValue<<<blocks, threads>>>(d_dest, nxyz, 1.0, d_src1, 1.0, d_src2);
	KCHECK
}

void cuda_addArrays(double* d_dest, int n, const double* d_src1, const double* d_src2)
{
	const int threads = 256;
	const int blocks = n / threads + 1;

	scaleaddValue<<<blocks, threads>>>(d_dest, n, 1.0, d_src1, 1.0, d_src2);
	KCHECK
}

void cuda_scaledAddArrays(double* d_dest, int n, const double s1, const double* d_src1, const double s2, const double* d_src2)
{
	const int threads = 256;
	const int blocks = n / threads + 1;

	scaleaddValue<<<blocks, threads>>>(d_dest, n, s1, d_src1, s2, d_src2);
	KCHECK
}


void ss_d_scaleadd3DArray(double* d_dest, int nxyz, double s1, double* d_src1, double s2, double* d_src2)
{
	const int threads = 256;
	const int blocks = nxyz / threads + 1;

	scaleaddValue<<<blocks, threads>>>(d_dest, nxyz, s1, d_src1, s2, d_src2);
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



// template <unsigned int blockSize>
// __device__ void warpReduce(volatile double *sdata, unsigned int tid)
// {
// 	if (blockSize >=  64) sdata[tid] += sdata[tid + 32];
// 	if (blockSize >=  32) sdata[tid] += sdata[tid + 16];
// 	if (blockSize >=  16) sdata[tid] += sdata[tid +  8];
// 	if (blockSize >=   8) sdata[tid] += sdata[tid +  4];
// 	if (blockSize >=   4) sdata[tid] += sdata[tid +  2];
// 	if (blockSize >=   2) sdata[tid] += sdata[tid +  1];
// }


template <unsigned int blockSize>
__global__ void reduce_sum_kernel(double *g_odata, double *g_idata, unsigned int n)
{
	__shared__ double sdata[blockSize];

	// each block of threads will work on blocksize^2 elements
	unsigned int work = blockSize * blockSize;
	
	unsigned int base = blockIdx.x * work;
	unsigned int tid = threadIdx.x;
	double mysum = 0;
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

#if 0
// template <unsigned int blockSize>
// __global__ void reduce_sum_kernel(double *g_odata, double *g_idata, unsigned int n)
// {
// 	__shared__ double sdata[blockSize];
// 	unsigned int tid = threadIdx.x;
// 	unsigned int i = blockIdx.x*(blockSize*2) + tid;
// 	unsigned int gridSize = blockSize*2*gridDim.x;
// 	sdata[tid] = 0;
// 	while (i < n)
// 	{
// 		sdata[tid] += g_idata[i] + g_idata[i+blockSize];
// 		i += gridSize;
// 	}
// 
// 	__syncthreads();
// 	if (blockSize >= 512)
// 	{
// 		if (tid < 256)
// 		{
// 			sdata[tid] += sdata[tid + 256];
// 		}
// 	__syncthreads();
// 	}
// 	
// 	if (blockSize >= 256)
// 	{
// 		if (tid < 128)
// 		{
// 			sdata[tid] += sdata[tid + 128];
// 		}
// 		__syncthreads();
// 	}
// 	if (blockSize >= 128)
// 	{
// 		if (tid <   64)
// 		{
// 			sdata[tid] += sdata[tid +   64];
// 		}
// 		__syncthreads();
// 	}
// 
// 	if(tid < 32)
// 		warpReduce<blockSize>(sdata, tid);
// 	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
// }
#endif

#define BS 64
double ss_reduce3DArray_sum(double* d_v, double* d_ws1, double* h_ws1, int nx, int ny, int nz)
{
	const unsigned int n = nx*ny*nz;
	const int work = BS*BS;
	const int blocks = 1 + n / work;
	const int threads = BS;
	
// 	int blocks = (nx*ny*nz) / (BS*BS) + 1;
#if 0
	double* d_d;
	double* h_d;
	
	printf("w %i    b %i   t %i\n", work, blocks, threads);
	
	malloc_device(&d_d, sizeof(double)*8);
	malloc_host(&h_d, sizeof(double)*8);
	
	h_d[0] = 1;
	h_d[1] = 3;
	h_d[2] = 1;
	h_d[3] = 1;
	h_d[4] = 5;
	h_d[5] = 4;
	h_d[6] = 1;
	h_d[7] = 1;
	
	memcpy_h2d(d_d, h_d, sizeof(double)*8);

	reduce_sum_kernel<BS><<<blocks, threads>>>(d_ws1, d_d, 8);
	KCHECK

	memcpy_d2h(h_d, d_ws1, sizeof(double)*8);

	for(int i=0; i<8; i++)
		printf("%3i %f\n", i, h_d[i]);
	
	free_host(h_d);
	free_device(d_d);
#endif
	
	
	
/*	
	
	reduce_sum_kernel<BS><<<blocks, threads>>>(d_ws1, d_d, 8);
	reduce_sum_kernel<BS><<<blocks, BS>>>(d_ws1, d_v, n);
	KCHECK

	CHECKCALL(cudaMemcpy(h_ws1, d_ws1, sizeof(double)*blocks, cudaMemcpyDeviceToHost));
	
*/

	reduce_sum_kernel<BS><<<blocks, threads>>>(d_ws1, d_v, n);
	KCHECK
// 	CHECKCALL(cudaMemcpy(h_ws1, d_ws1, sizeof(double)*blocks, cudaMemcpyDeviceToHost));
	
	
	memcpy_d2h(h_ws1, d_ws1, sizeof(double)*blocks);

	for(int i=1; i<blocks; i++)
		h_ws1[0] += h_ws1[i];
	
	return h_ws1[0];
}



void ss_d_copyArray(double* d_dest, double* d_src, int nxyz)
{
	cudaMemcpy(d_dest, d_src, sizeof(double)*nxyz, cudaMemcpyDeviceToDevice);
}


