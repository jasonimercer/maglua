#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>
#include "array_ops.hpp"
#include <stdio.h>
#include "memory.hpp"
#include "hd_helper_tfuncs.hpp"

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

#define THREAD_COUNT 256




template<typename T>
__global__ void setValue(T* dest, const int n, const T v)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= n)
		return;
	dest[idx] = v;
}

template<typename T>
void setAll_(T* d_dest, const int n, const T& v)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	setValue<T><<<blocks, threads>>>(d_dest, n, v);
	KCHECK
}

void arraySetAll(double* a, const int n, const double& v)
{
	setAll_<double>(a, n, v);
}
void arraySetAll(float* a, const int n, const float& v)
{
	setAll_<float>(a, n, v);
}
void arraySetAll(int* a, const int n, const int& v)
{
	setAll_<int>(a, n, v);
}
void arraySetAll(doubleComplex* a, const int n, const doubleComplex& v)
{
	setAll_<doubleComplex>(a, n, v);
}
void arraySetAll(floatComplex* a, const int n, const floatComplex& v)
{
	setAll_<floatComplex>(a, n, v);
}





// dest[i] = (mult1 * src1[i] + add1) * (mult2 * src2[i] + add2)
template<typename T>
__global__ void multi_op(       T* dest, const T* src1, T* src2, const int n,
						  const T mult1,  const T add1, 
						  const T mult2,  const T add2)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= n)
		return;
	
	T t1 = src2[idx];
	T t2 = src1[idx];

	times_equal(t1, mult1);
	plus_equal(t1, add1);
	
	times_equal(t2, mult2);
	plus_equal(t2, add2);
	
	times_equal(t1, t2);
	dest[idx] = t1;
}


template<typename T>
void scaleAll_(T* d_dest, const int n, const T& v)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	multi_op<T><<<blocks, threads>>>(d_dest, d_dest, d_dest, n,
									 v, zero<T>(), zero<T>(), one<T>()); 
	KCHECK
}


void arrayScaleAll(double* a, const int n, const double& v)
{
	scaleAll_<double>(a, n, v);
}
void arrayScaleAll(float* a, const int n, const float& v)
{
	scaleAll_<float>(a, n, v);
}
void arrayScaleAll(int* a, const int n, const int& v)
{
	scaleAll_<int>(a, n, v);
}
void arrayScaleAll(doubleComplex* a, const int n, const doubleComplex& v)
{
	scaleAll_<doubleComplex>(a, n, v);
}
void arrayScaleAll(floatComplex* a, const int n, const floatComplex& v)
{
	scaleAll_<floatComplex>(a, n, v);
}





// dest[i] = (mult1 * src1[i] + add1) * (mult2 * src2[i] + add2)
template<typename T>
void arrayDot_(T* d_dest, T* d_s1, T* d_s2, const int n)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	multi_op<T><<<blocks, threads>>>(d_dest, d_s1, d_s2, n,
									 one<T>(), zero<T>(), one<T>(), zero<T>()); 
	KCHECK
}


void arrayMultAll(double* d, double* s1, double* s2, const int n)
{
	arrayDot_<double>(d, s1, s2, n);
}
void arrayMultAll(float* d, float* s1, float* s2, const int n)
{
	arrayDot_<float>(d, s1, s2, n);
}
void arrayMultAll(int* d, int* s1, int* s2, const int n)
{
	arrayDot_<int>(d, s1, s2, n);
}
void arrayMultAll(doubleComplex* d, doubleComplex* s1, doubleComplex* s2, const int n)
{
	arrayDot_<doubleComplex>(d, s1, s2, n);
}
void arrayMultAll(floatComplex* d, floatComplex* s1, floatComplex* s2, const int n)
{
	arrayDot_<floatComplex>(d, s1, s2, n);
}








// dest[i] = (mult1 * src1[i] + add1) * (mult2 * src2[i] + add2)
template<typename T>
void arrayDiff_(T* d_dest, T* d_s1, T* d_s2, const int n)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	multi_op<T><<<blocks, threads>>>(d_dest, d_s1, d_s2, n,
									 one<T>(), zero<T>(), negone<T>(), zero<T>()); 
	KCHECK
}

void arrayDiffAll(double* d, double* s1, double* s2, const int n)
{
	arrayDiff_<double>(d, s1, s2, n);
}
void arrayDiffAll(float* d, float* s1, float* s2, const int n)
{
	arrayDiff_<float>(d, s1, s2, n);
}
void arrayDiffAll(int* d, int* s1, int* s2, const int n)
{
	arrayDiff_<int>(d, s1, s2, n);
}
void arrayDiffAll(doubleComplex* d, doubleComplex* s1, doubleComplex* s2, const int n)
{
	arrayDiff_<doubleComplex>(d, s1, s2, n);
}
void arrayDiffAll(floatComplex* d, floatComplex* s1, floatComplex* s2, const int n)
{
	arrayDiff_<floatComplex>(d, s1, s2, n);
}








template<typename T>
__global__ void norm_op(T* dest, const T* src, const int n)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= n)
		return;
	
	set_norm(dest[idx], src[idx]);
}

// dest[i] = (mult1 * src1[i] + add1) * (mult2 * src2[i] + add2)
template<typename T>
void arrayNorm_(T* d_dest, T* d_s1, const int n)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	norm_op<T><<<blocks, threads>>>(d_dest, d_s1, n); 
	KCHECK
}

void arrayNormAll(double* d, double* s1, const int n)
{
	arrayNorm_<double>(d, s1, n);
}
void arrayNormAll(float* d, float* s1, const int n)
{
	arrayNorm_<float>(d, s1, n);
}
void arrayNormAll(int* d, int* s1, const int n)
{
	arrayNorm_<int>(d, s1, n);
}
void arrayNormAll(doubleComplex* d, doubleComplex* s1, const int n)
{
	arrayNorm_<doubleComplex>(d, s1, n);
}
void arrayNormAll(floatComplex* d, floatComplex* s1, const int n)
{
	arrayNorm_<floatComplex>(d, s1, n);
}




template<typename A, typename B>
__global__ void real_op(A* d_dest, const B* d_src, const int n)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= n)
		return;
	d_dest[idx] = d_src[idx].x;
}
template<typename A, typename B>
__global__ void imag_op(A* d_dest, const B* d_src, const int n)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= n)
		return;
	d_dest[idx] = d_src[idx].y;
}

template<typename A, typename B>
void arrayGetRealPart_(A* d_dest, const B* d_src, const int n)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	real_op<A,B><<<blocks, threads>>>(d_dest, d_src, n); 
	KCHECK
}
template<typename A, typename B>
void arrayGetImagPart_(A* d_dest, const B* d_src, const int n)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	imag_op<A,B><<<blocks, threads>>>(d_dest, d_src, n); 
	KCHECK
}




void arrayGetRealPart(double* dest, const doubleComplex* src, const int n)
{
	arrayGetRealPart_<double,doubleComplex>(dest, src, n);
}
void arrayGetRealPart(float* dest, const floatComplex* src, const int n)
{
	arrayGetRealPart_<float,floatComplex>(dest, src, n);
}

void arrayGetImagPart(double* dest, const doubleComplex* src, const int n)
{
	arrayGetImagPart_<double,doubleComplex>(dest, src, n);
}
void arrayGetImagPart(float* dest, const floatComplex* src, const int n)
{
	arrayGetImagPart_<float,floatComplex>(dest, src, n);
}




template<typename A, typename B>
__global__ void set_real_op(A* d_dest, const B* d_src, const int n)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= n)
		return;
	d_dest[idx].x = d_src[idx];
}
template<typename A, typename B>
__global__ void set_imag_op(A* d_dest, const B* d_src, const int n)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(idx >= n)
		return;
	d_dest[idx].y = d_src[idx];
}


template<typename A, typename B>
void arraySetRealPart_(A* d_dest, const B* d_src, const int n)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	set_real_op<A,B><<<blocks, threads>>>(d_dest, d_src, n); 
	KCHECK
}
template<typename A, typename B>
void arraySetImagPart_(A* d_dest, const B* d_src, const int n)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	set_imag_op<A,B><<<blocks, threads>>>(d_dest, d_src, n); 
	KCHECK
}



void arraySetRealPart(doubleComplex* d_dest, const double * d_src, const int n)
{
	arraySetRealPart_<doubleComplex,double>(d_dest, d_src, n);
}
void arraySetRealPart(floatComplex* d_dest, const float * d_src, const int n)
{
	arraySetRealPart_<floatComplex,float>(d_dest, d_src, n);
}

void arraySetImagPart(doubleComplex* d_dest, const double * d_src, const int n)
{
	arraySetImagPart_<doubleComplex,double>(d_dest, d_src, n);
}
void arraySetImagPart(floatComplex* d_dest, const float * d_src, const int n)
{
	arraySetImagPart_<floatComplex,float>(d_dest, d_src, n);
}





template <unsigned int blockSize, typename T>
__global__ void reduce_sum_kernel(T *g_odata, T *g_idata, unsigned int n)
{
	__shared__ T sdata[blockSize];

	// each block of threads will work on blocksize^2 elements
	unsigned int work = blockSize * blockSize;
	
	unsigned int base = blockIdx.x * work;
	unsigned int tid = threadIdx.x;
	T mysum = zero<T>();
	
#pragma unroll
	for(int j=0; j<blockSize; j++)
	{
		const int k = base + j*blockSize + tid;
		if(k < n) plus_equal<T>(mysum, g_idata[k]);
	}
	sdata[tid] = mysum;
	__syncthreads();

// #pragma unroll
	for(int j=2; j<=blockSize; j*=2)
	{
		if(tid < blockSize/j)// < n/(j-1))
			plus_equal<T>(sdata[tid], sdata[tid + blockSize/j]);
		__syncthreads();
	}
	
	if(tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

#define BS 64
template<typename T>
T arraySumAll_(T* d_v, const int n)
{
	const int work = BS*BS;
	const int blocks = 1 + n / work;
	const int threads = BS;

	T* d_ws1;
	T* h_ws1;
	
	malloc_device(&d_ws1, sizeof(T)*n);
	malloc_host(&h_ws1, sizeof(T)*n);
	
	reduce_sum_kernel<BS><<<blocks, threads>>>(d_ws1, d_v, n);
	KCHECK
	
	memcpy_d2h(h_ws1, d_ws1, sizeof(T)*blocks);

	for(int i=1; i<blocks; i++)
		plus_equal<T>(h_ws1[0], h_ws1[i]);

	T res = h_ws1[0];
	
	free_device(d_ws1);
	free_host(h_ws1);
	
	return res;
}



void arraySumAll(double* a, const int n, double& v)
{
	v = arraySumAll_<double>(a, n);
}
void arraySumAll(float* a, const int n, float& v)
{
	v = arraySumAll_<float>(a, n);
}
void arraySumAll(int* a, const int n, int& v)
{
	v = arraySumAll_<int>(a, n);
}
void arraySumAll(doubleComplex* a, const int n, doubleComplex& v)
{
	v = arraySumAll_<doubleComplex>(a, n);
}
void arraySumAll(floatComplex* a, const int n, floatComplex& v)
{
	v = arraySumAll_<floatComplex>(a, n);
}







