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
__global__ void arraySetAll__(T* dest, const T v, const int n)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= n)
		return;
	dest[idx] = v;
}

template<typename T>
void arraySetAll_(T* a, const T& v, const int n)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	arraySetAll__<T><<<blocks, threads>>>(a, v, n);
	KCHECK
}

void arraySetAll(double* d_a, const double& v, const int n)
{
	arraySetAll_<double>(d_a, v, n);
}
void arraySetAll(float* d_a,  const float& v, const int n)
{
	arraySetAll_<float>(d_a, v, n);
}
void arraySetAll(int* d_a, const int& v, const int n)
{
	arraySetAll_<int>(d_a, v, n);
}
void arraySetAll(doubleComplex* d_a, const doubleComplex& v, const int n)
{
	arraySetAll_<doubleComplex>(d_a, v, n);
}
void arraySetAll(floatComplex* d_a,  const floatComplex& v, const int n)
{
	arraySetAll_<floatComplex>(d_a, v, n);
}






template<typename T>
__global__ void arrayScaleAll_o__(T* d, const T v, const int n, const int offset)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= n)
		return;

	times_equal(d[idx + offset], v);
}


template<typename T>
void arrayScaleAll_o_(T* d_dest, const T& v, const int n, const int offset=0)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	arrayScaleAll_o__<T><<<blocks, threads>>>(d_dest, v, n, offset);
	KCHECK
}

void arrayScaleAll(double* d_a, const double& v, const int n)
{
	arrayScaleAll_o_<double>(d_a, v, n);
}
void arrayScaleAll(float* d_a, const float& v, const int n)
{
	arrayScaleAll_o_<float>(d_a, v, n);
}
void arrayScaleAll(int* d_a, const int& v, const int n)
{
	arrayScaleAll_o_<int>(d_a, v, n);
}
void arrayScaleAll(doubleComplex* d_a, const doubleComplex& v, const int n)
{
	arrayScaleAll_o_<doubleComplex>(d_a, v, n);
}
void arrayScaleAll(floatComplex* d_a, const floatComplex& v, const int n)
{
	arrayScaleAll_o_<floatComplex>(d_a, v, n);
}




void arrayScaleAll_o(double* d_a, const int offset, const double& v, const int n)
{
	arrayScaleAll_o_<double>(d_a, v, n, offset);
}
void arrayScaleAll_o(float* d_a, const int offset, const float& v, const int n)
{
	arrayScaleAll_o_<float>(d_a, v, n, offset);
}
void arrayScaleAll_o(int* d_a, const int offset, const int& v, const int n)
{
	arrayScaleAll_o_<int>(d_a, v, n, offset);
}
void arrayScaleAll_o(doubleComplex* d_a, const int offset, const doubleComplex& v, const int n)
{
	arrayScaleAll_o_<doubleComplex>(d_a, v, n, offset);
}
void arrayScaleAll_o(floatComplex* d_a, const int offset, const floatComplex& v, const int n)
{
	arrayScaleAll_o_<floatComplex>(d_a, v, n, offset);
}





template<typename T>
__global__ void arrayAddAll_o__(T* d, const T v, const int n, const int offset)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= n)
		return;

	plus_equal(d[idx + offset], v);
}
template<typename T>
void arrayAddAll_o_(T* d_dest, const T& v, const int n, const int offset=0)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	arrayAddAll_o__<T><<<blocks, threads>>>(d_dest, v, n, offset);
	KCHECK
}
void arrayAddAll(double* d_a, const double& v, const int n)
{
	arrayScaleAll_o_<double>(d_a, v, n);
}
void arrayAddAll(float* d_a, const float& v, const int n)
{
	arrayScaleAll_o_<float>(d_a, v, n);
}
void arrayAddAll(int* d_a, const int& v, const int n)
{
	arrayScaleAll_o_<int>(d_a, v, n);
}
void arrayAddAll(doubleComplex* d_a, const doubleComplex& v, const int n)
{
	arrayScaleAll_o_<doubleComplex>(d_a, v, n);
}
void arrayAddAll(floatComplex* d_a, const floatComplex& v, const int n)
{
	arrayScaleAll_o_<floatComplex>(d_a, v, n);
}




template<typename T>
__global__ void arrayMultAll__(T* d_dest, T* d_src1, T* d_src2, const int n)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= n)
		return;

	T t = d_src1[idx];
	times_equal(t, d_src2[idx]);
	d_dest[idx] = t;
}
template<typename T>
void arrayMultAll_(T* d_dest, T* d_src1, T* d_src2, const int n)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	arrayMultAll__<T><<<blocks, threads>>>(d_dest, d_src1, d_src2, n);
	KCHECK
}
void arrayMultAll(double* d_dest, double* d_src1, double* d_src2, const int n)
{
	arrayMultAll_<double>(d_dest, d_src1, d_src2, n);
}
void arrayMultAll(float* d_dest, float* d_src1, float* d_src2, const int n)
{
	arrayMultAll_<float>(d_dest, d_src1, d_src2, n);
}
void arrayMultAll(int* d_dest, int* d_src1, int* d_src2, const int n)
{
	arrayMultAll_<int>(d_dest, d_src1, d_src2, n);
}
void arrayMultAll(doubleComplex* d_dest, doubleComplex* d_src1, doubleComplex* d_src2, const int n)
{
	arrayMultAll_<doubleComplex>(d_dest, d_src1, d_src2, n);
}
void arrayMultAll(floatComplex* d_dest, floatComplex* d_src1, floatComplex* d_src2, const int n)
{
	arrayMultAll_<floatComplex>(d_dest, d_src1, d_src2, n);
}



template<typename T>
__global__ void arrayDiffAll__(T* d_dest, T* d_src1, T* d_src2, const int n)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= n)
		return;

	T t1 = d_src1[idx];
	T t2 = d_src2[idx];
	times_equal(t2, negone<T>());
	plus_equal(t1, t2);
	d_dest[idx] = t1;
}
template<typename T>
void arrayDiffAll_(T* d_dest, T* d_src1, T* d_src2, const int n)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	arrayDiffAll__<T><<<blocks, threads>>>(d_dest, d_src1, d_src2, n);
	KCHECK
}
void arrayDiffAll(double* d_dest, double* d_src1, double* d_src2, const int n)
{
	arrayDiffAll_<double>(d_dest, d_src1, d_src2, n);
}
void arrayDiffAll(float* d_dest, float* d_src1, float* d_src2, const int n)
{
	arrayDiffAll_<float>(d_dest, d_src1, d_src2, n);
}
void arrayDiffAll(int* d_dest, int* d_src1, int* d_src2, const int n)
{
	arrayDiffAll_<int>(d_dest, d_src1, d_src2, n);
}
void arrayDiffAll(doubleComplex* d_dest, doubleComplex* d_src1, doubleComplex* d_src2, const int n)
{
	arrayDiffAll_<doubleComplex>(d_dest, d_src1, d_src2, n);
}
void arrayDiffAll(floatComplex* d_dest, floatComplex* d_src1, floatComplex* d_src2, const int n)
{
	arrayDiffAll_<floatComplex>(d_dest, d_src1, d_src2, n);
}





template<typename T>
__global__ void arraySumAll__(T* d_dest, const T* d_src1, const T* d_src2, const int n)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= n)
		return;

	T t1 = d_src1[idx];
	const T t2 = d_src2[idx];
	plus_equal(t1, t2);
	d_dest[idx] = t1;
}
template<typename T>
void arraySumAll_(T* d_dest, const T* d_src1, const T* d_src2, const int n)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	arraySumAll__<T><<<blocks, threads>>>(d_dest, d_src1, d_src2, n);
	KCHECK
}
void arraySumAll(double* d_dest, const double* d_src1, const double* d_src2, const int n)
{
	arraySumAll_<double>(d_dest, d_src1, d_src2, n);
}
void arraySumAll(float* d_dest, const float* d_src1, const float* d_src2, const int n)
{
	arraySumAll_<float>(d_dest, d_src1, d_src2, n);
}
void arraySumAll(int* d_dest, const int* d_src1, const int* d_src2, const int n)
{
	arraySumAll_<int>(d_dest, d_src1, d_src2, n);
}
void arraySumAll(doubleComplex* d_dest, const doubleComplex* d_src1, const doubleComplex* d_src2, const int n)
{
	arraySumAll_<doubleComplex>(d_dest, d_src1, d_src2, n);
}
void arraySumAll(floatComplex* d_dest, const floatComplex* d_src1, const floatComplex* d_src2, const int n)
{
	arraySumAll_<floatComplex>(d_dest, d_src1, d_src2, n);
}






template<typename T>
__global__ void arrayNormAll__(T* d_dest, T* d_src1, const int n)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= n)
		return;

	set_norm(d_dest[idx], d_src1[idx]);
}
template<typename T>
void arrayNormAll_(T* d_dest, T* d_src1, const int n)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	arrayNormAll__<T><<<blocks, threads>>>(d_dest, d_src1, n);
	KCHECK
}
void arrayNormAll(double* d_dest, double* d_src1, const int n)
{
	arrayNormAll_<double>(d_dest, d_src1, n);
}
void arrayNormAll(float* d_dest, float* d_src1, const int n)
{
	arrayNormAll_<float>(d_dest, d_src1, n);
}
void arrayNormAll(int* d_dest, int* d_src1, const int n)
{
	arrayNormAll_<int>(d_dest, d_src1, n);
}
void arrayNormAll(doubleComplex* d_dest, doubleComplex* d_src1, const int n)
{
	arrayNormAll_<doubleComplex>(d_dest, d_src1, n);
}
void arrayNormAll(floatComplex* d_dest, floatComplex* d_src1, const int n)
{
	arrayNormAll_<floatComplex>(d_dest, d_src1, n);
}







template <unsigned int blockSize, typename T>
__global__ void reduceSumAll__(T *g_odata, const T *g_idata, unsigned int n)
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
T reduceSumAll_(const T* d_v, const int n)
{
	const int work = BS*BS;
	const int blocks = 1 + n / work;
	const int threads = BS;

	T* d_ws1;
	T* h_ws1;

	malloc_dh(&d_ws1,&h_ws1, sizeof(T)*n);

	reduceSumAll__<BS><<<blocks, threads>>>(d_ws1, d_v, n);
	KCHECK

	memcpy_d2h(h_ws1, d_ws1, sizeof(T)*blocks);

	for(int i=1; i<blocks; i++)
		plus_equal<T>(h_ws1[0], h_ws1[i]);

	T res = h_ws1[0];

	free_dh(d_ws1, h_ws1);

	return res;
}
#undef BS
void reduceSumAll(const double* a, const int n, double& v)
{
	v = reduceSumAll_<double>(a, n);
}
void reduceSumAll(const float* a, const int n, float& v)
{
	v = reduceSumAll_<float>(a, n);
}
void reduceSumAll(const int* a, const int n, int& v)
{
	v = reduceSumAll_<int>(a, n);
}
void reduceSumAll(const doubleComplex* a, const int n, doubleComplex& v)
{
	v = reduceSumAll_<doubleComplex>(a, n);
}
void reduceSumAll(const floatComplex* a, const int n, floatComplex& v)
{
	v = reduceSumAll_<floatComplex>(a, n);
}















template <unsigned int blockSize, typename T>
__global__ void reduceDiffSumAll__(T *g_odata, const T *g_idata1, const T *g_idata2, unsigned int n)
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
		if(k < n)
		{
			T a = g_idata1[k];
			T b = g_idata2[k];

			times_equal(b,  negone<T>());
			plus_equal(a, b);
			set_norm(b, a);
			plus_equal<T>(mysum, b);
		}
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
T reduceDiffSumAll_(const T* a, const T* b, const int n)
{
	const int work = BS*BS;
	const int blocks = 1 + n / work;
	const int threads = BS;

	T* d_ws1;
	T* h_ws1;

	malloc_dh(&d_ws1,&h_ws1, sizeof(T)*n);

	reduceDiffSumAll__<BS><<<blocks, threads>>>(d_ws1, a, b, n);
	KCHECK

	memcpy_d2h(h_ws1, d_ws1, sizeof(T)*blocks);

	for(int i=1; i<blocks; i++)
		plus_equal<T>(h_ws1[0], h_ws1[i]);

	T res = h_ws1[0];

	free_dh(d_ws1, h_ws1);

	return res;
}
#undef BS
void reduceDiffSumAll(const double* a, const double* b, const int n, double& v)
{
	v = reduceDiffSumAll_<double>(a, b, n);
}
void reduceDiffSumAll(const float* a, const float* b, const int n, float& v)
{
	v = reduceDiffSumAll_<float>(a, b, n);
}
void reduceDiffSumAll(const int* a, const int* b, const int n, int& v)
{
	v = reduceDiffSumAll_<int>(a, b, n);
}
void reduceDiffSumAll(const doubleComplex* a, const doubleComplex* b, const int n, doubleComplex& v)
{
	v = reduceDiffSumAll_<doubleComplex>(a, b, n);
}
void reduceDiffSumAll(const floatComplex* a, const floatComplex* b, const int n, floatComplex& v)
{
	v = reduceDiffSumAll_<floatComplex>(a, b, n);
}








template <unsigned int blockSize, typename T>
__global__ void reduceMultSumAll__(T *g_odata, const T *g_idata1, const T *g_idata2, unsigned int n)
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
		if(k < n)
		{
			T a = g_idata1[k];
			const T b = g_idata2[k];

			times_equal(a, b);
			plus_equal<T>(mysum, a);
		}
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
T reduceMultSumAll_(const T* a, const T* b, const int n)
{
	const int work = BS*BS;
	const int blocks = 1 + n / work;
	const int threads = BS;

	T* d_ws1;
	T* h_ws1;

	malloc_dh(&d_ws1,&h_ws1, sizeof(T)*n);

	reduceMultSumAll__<BS><<<blocks, threads>>>(d_ws1, a, b, n);
	KCHECK

	memcpy_d2h(h_ws1, d_ws1, sizeof(T)*blocks);

	for(int i=1; i<blocks; i++)
		plus_equal<T>(h_ws1[0], h_ws1[i]);

	T res = h_ws1[0];

	free_dh(d_ws1, h_ws1);

	return res;
}
#undef BS
void reduceMultSumAll(const double* a, const double* b, const int n, double& v)
{
	v = reduceMultSumAll_<double>(a, b, n);
}
void reduceMultSumAll(const float* a, const float* b, const int n, float& v)
{
	v = reduceMultSumAll_<float>(a, b, n);
}
void reduceMultSumAll(const int* a, const int* b, const int n, int& v)
{
	v = reduceMultSumAll_<int>(a, b, n);
}
void reduceMultSumAll(const doubleComplex* a, const doubleComplex* b, const int n, doubleComplex& v)
{
	v = reduceMultSumAll_<doubleComplex>(a, b, n);
}
void reduceMultSumAll(const floatComplex* a, const floatComplex* b, const int n, floatComplex& v)
{
	v = reduceMultSumAll_<floatComplex>(a, b, n);
}







template <unsigned int blockSize, unsigned int do_max, typename T>
__global__ void reduceExtreme__(const T *idata, T *odata, int* oidx, const int n)
{
	__shared__ T   sdata[blockSize];
	__shared__ int sidx[blockSize];

	// each block of threads will work on blocksize^2 elements
	unsigned int work = blockSize * blockSize;

	unsigned int base = blockIdx.x * work;
	unsigned int tid = threadIdx.x;

	int my_idx = base + tid;
	T   my_extreme = idata[my_idx];

	if(do_max) //hope this gets optimized
	{	
		#pragma unroll
		for(int j=0; j<blockSize; j++)
		{
			const int k = base + j*blockSize + tid;
			if(k < n)
			{
				const T a = idata[k];

				if(less_than<T>(my_extreme, a))
				//if(a > my_extreme)
				{
					my_idx  = k;
					my_extreme = a;
				}
			}
		}
	}
	else
	{
		#pragma unroll
		for(int j=0; j<blockSize; j++)
		{
			const int k = base + j*blockSize + tid;
			if(k < n)
			{
				const T a = idata[k];


				if(less_than<T>(a, my_extreme))
				//if(a < my_extreme)
				{
					my_idx  = k;
					my_extreme = a;
				}
			}
		}
	}
	sdata[tid] = my_extreme;
	sidx[tid] = my_idx;
	__syncthreads();

// #pragma unroll
	for(int j=2; j<=blockSize; j*=2)
	{
		if(tid < blockSize/j)// < n/(j-1))
		{
			if(do_max)
			{
				if(less_than<T>(sdata[tid], sdata[tid + blockSize/j]))
				//if(sdata[tid] < sdata[tid + blockSize/j])
				{
					sdata[tid] = sdata[tid + blockSize/j];
					sidx[tid] = sidx[tid + blockSize/j];
				}
			}
			else
			{
				if(less_than<T>(sdata[tid + blockSize/j], sdata[tid]))
				//if(sdata[tid] > sdata[tid + blockSize/j])
				{
					sdata[tid] = sdata[tid + blockSize/j];
					sidx[tid] = sidx[tid + blockSize/j];
				}
			}
		}
		__syncthreads();
	}

	if(tid == 0)
	{
		odata[blockIdx.x] = sdata[0];
		oidx[blockIdx.x] = sidx[0];
	}
}

#define BS 64
template<typename T>
T reduceExtreme_(const T* d_a, const int min_max, const int n, T& v, int& idx)
{
	const int work = BS*BS;
	const int blocks = 1 + n / work;
	const int threads = BS;

	T* d_ws1;
	T* h_ws1;

	int* d_idx;
	int* h_idx;

	malloc_dh(&d_ws1,&h_ws1, sizeof(T)*n);
	malloc_dh(&d_idx,&h_idx, sizeof(int)*n);

	if(min_max)
		reduceExtreme__<BS, 1, T><<<blocks, threads>>>(d_a, d_ws1, d_idx, n);
	else
		reduceExtreme__<BS, 0, T><<<blocks, threads>>>(d_a, d_ws1, d_idx, n);
	KCHECK

	memcpy_d2h(h_ws1, d_ws1, sizeof(T)*blocks);
	memcpy_d2h(h_idx, d_idx, sizeof(int)*n);


	T res_val = h_ws1[0];
	int res_idx = h_idx[0];
	if(min_max)
	{
		for(int i=1; i<blocks; i++)
		{
			if(less_than<T>(res_val, h_ws1[i]))
// 			if(h_ws1[i] > res_val)
			{
				res_val = h_ws1[i];
				res_idx = h_idx[i];
			}
		}
	}
	else
	{
		for(int i=1; i<blocks; i++)
		{
			if(less_than<T>(h_ws1[i], res_val))
			//if(h_ws1[i] < res_val)
			{
				res_val = h_ws1[i];
				res_idx = h_idx[i];
			}
		}
	}

	free_dh(d_ws1, h_ws1);
	free_dh(d_idx, h_idx);

	v = res_val;
	idx = res_idx;

	return v;
}
#undef BS
void reduceExtreme(const double* d_a, const int min_max, const int n, double& v, int& idx)
{
	v = reduceExtreme_<double>(d_a, min_max, n, v, idx);
}
void reduceExtreme(const float* d_a, const int min_max, const int n, float& v, int& idx)
{
	v = reduceExtreme_<float>(d_a, min_max, n, v, idx);
}
void reduceExtreme(const int* d_a, const int min_max, const int n, int& v, int& idx)
{
	v = reduceExtreme_<int>(d_a, min_max, n, v, idx);
}
void reduceExtreme(const doubleComplex* d_a, const int min_max, const int n, doubleComplex& v, int& idx)
{
	v = reduceExtreme_<doubleComplex>(d_a, min_max, n, v, idx);
}
void reduceExtreme(const floatComplex* d_a, const int min_max, const int n, floatComplex& v, int& idx)
{
	v = reduceExtreme_<floatComplex>(d_a, min_max, n, v, idx);
}



template<typename T>
__global__ void arrayScaleAdd__(T* dest, T s1, const T* src1, const T s2, const T* src2, const int n)

{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= n)
		return;

	T a = s1;
	times_equal(a, src1[idx]);
	T b = s2;
	times_equal(b, src2[idx]);

	plus_equal(a, b);

	dest[idx] = a;
}
template<typename T>
void arrayScaleAdd_(T* dest, T s1, const T* src1, const T s2, const T* src2, const int n)
{
	const int threads = THREAD_COUNT;
	const int blocks = n / threads + 1;

	arrayScaleAdd__<T><<<blocks, threads>>>(dest, s1, src1, s2, src2, n);
	KCHECK
}

void arrayScaleAdd(double* dest, double s1, const double* src1, const double s2, const double* src2, const int n)
{
	arrayScaleAdd_<double>(dest, s1, src1, s2, src2, n);
}

void arrayScaleAdd(float* dest, float s1, const float* src1, float s2, const float* src2, const int n)
{
	arrayScaleAdd_<float>(dest, s1, src1, s2, src2, n);
}
void arrayScaleAdd(int* dest, int s1, const int* src1, int s2, const int* src2, const int n)
{
	arrayScaleAdd_<int>(dest, s1, src1, s2, src2, n);
}
void arrayScaleAdd(doubleComplex* dest, doubleComplex s1, const doubleComplex* src1, doubleComplex s2, const doubleComplex* src2, const int n)
{
	arrayScaleAdd_<doubleComplex>(dest, s1, src1, s2, src2, n);
}
void arrayScaleAdd(floatComplex* dest, floatComplex s1, const floatComplex* src1, floatComplex s2, const floatComplex* src2, const int n)
{
	arrayScaleAdd_<floatComplex>(dest, s1, src1, s2, src2, n);
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










template<typename T>
__global__ void arrayLayerMultSet__(T* d, int dl, const T* s1, int s1l, const T* s2, int s2l, T mult, const int nxy)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= nxy)
		return;

	d[idx + dl*nxy] = s1[idx + s1l*nxy];
	times_equal(d[idx + dl*nxy], s2[idx + s2l*nxy]);
	times_equal(d[idx + dl*nxy], mult);
}

template<typename T>
__global__ void arrayLayerMultTotal__(T* d, int dl, const T* s1, int s1l, const T* s2, int s2l, T mult, const int nxy)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= nxy)
		return;

	T t = d[idx + dl*nxy];
	d[idx + dl*nxy] = s1[idx + s1l*nxy];
	times_equal(d[idx + dl*nxy], s2[idx + s2l*nxy]);
	times_equal(d[idx + dl*nxy], mult);
	plus_equal(d[idx + dl*nxy] , t);
}

template<typename T>
void arrayLayerMult_(T* d, int dl, const T* s1, int s1l, const T* s2, int s2l, T mult, const int set, const int nxy)
{
	const int threads = THREAD_COUNT;
	const int blocks = nxy / threads + 1;

	if(set)
		arrayLayerMultSet__<T><<<blocks, threads>>>(d, dl, s1, s1l, s2, s2l, mult, nxy);
	else
		arrayLayerMultTotal__<T><<<blocks, threads>>>(d, dl, s1, s1l, s2, s2l, mult, nxy);
	KCHECK
}

void arrayLayerMult(double* dest, int dest_layer, const double* src1, int src1_layer, const double* src2, int src2_layer, double mult, int set, const int nxy)
{
	arrayLayerMult_<double>(dest, dest_layer, src1, src1_layer, src2, src2_layer, mult, set, nxy);
}
void arrayLayerMult(float* dest, int dest_layer, const float* src1, int src1_layer, const float* src2, int src2_layer, float mult, int set, const int nxy)
{
	arrayLayerMult_<float>(dest, dest_layer, src1, src1_layer, src2, src2_layer, mult, set, nxy);
}
void arrayLayerMult(int* dest, int dest_layer, const int* src1, int src1_layer, const int* src2, int src2_layer, int mult, int set, const int nxy)
{
	arrayLayerMult_<int>(dest, dest_layer, src1, src1_layer, src2, src2_layer, mult, set, nxy);
}
void arrayLayerMult(doubleComplex* dest, int dest_layer, const doubleComplex* src1, int src1_layer, const doubleComplex* src2, int src2_layer, doubleComplex mult, int set, const int nxy)
{
	arrayLayerMult_<doubleComplex>(dest, dest_layer, src1, src1_layer, src2, src2_layer, mult, set, nxy);
}
void arrayLayerMult(floatComplex* dest, int dest_layer, const floatComplex* src1, int src1_layer, const floatComplex* src2, int src2_layer, floatComplex mult, int set, const int nxy)
{
	arrayLayerMult_<floatComplex>(dest, dest_layer, src1, src1_layer, src2, src2_layer, mult, set, nxy);
}









template<typename T>
__global__ void arrayScaleMultAdd__(T* dest, const int od, T scale, const T* src1, const int o1, const T* src2, const int o2, const T* src3, const int o3, const int nxy)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if(idx >= nxy)
		return;

	T a = src1[idx+o1];
	times_equal(a, src2[idx+o2]);
	times_equal(a, scale);
	plus_equal(a, src3[idx+o3]);

	dest[idx + od] = a;
}
template<typename T>
void arrayScaleMultAdd_(T* dest, const int od, T scale, const T* src1, const int o1, const T* src2, const int o2, const T* src3, const int o3, const int nxy)
{
	const int threads = THREAD_COUNT;
	const int blocks = nxy / threads + 1;

	arrayScaleMultAdd__<T><<<blocks, threads>>>(dest, od, scale, src1, o1, src2, o2, src3, o3, nxy);
	KCHECK
}

// _o arbitrarily means offset
void arrayScaleMultAdd_o(double* dest, const int od, double scale, const double* src1, const int o1, const double* src2, const int o2, const double* src3, const int o3, const int nxy)
{
	arrayScaleMultAdd_<double>(dest, od, scale, src1, o1, src2, o2, src3, o3, nxy);
}

void arrayScaleMultAdd_o(float* dest, const int od, float scale, const float* src1, const int o1, const float* src2, const int o2, const float* src3, const int o3, const int nxy)
{
	arrayScaleMultAdd_<float>(dest, od, scale, src1, o1, src2, o2, src3, o3, nxy);
}

void arrayScaleMultAdd_o(int* dest, const int od, int scale, const int* src1, const int o1, const int* src2, const int o2, const int* src3, const int o3, const int nxy)
{
	arrayScaleMultAdd_<int>(dest, od, scale, src1, o1, src2, o2, src3, o3, nxy);
}

void arrayScaleMultAdd_o(doubleComplex* dest, const int od, doubleComplex scale, const doubleComplex* src1, const int o1, const doubleComplex* src2, const int o2, const doubleComplex* src3, const int o3, const int nxy)
{
	arrayScaleMultAdd_<doubleComplex>(dest, od, scale, src1, o1, src2, o2, src3, o3, nxy);
}

void arrayScaleMultAdd_o(floatComplex* dest, const int od, floatComplex scale, const floatComplex* src1, const int o1, const floatComplex* src2, const int o2, const floatComplex* src3, const int o3, const int nxy)
{
	arrayScaleMultAdd_<floatComplex>(dest, od, scale, src1, o1, src2, o2, src3, o3, nxy);
}



















	









template <unsigned int blockSize, typename T>
__global__ void arrayAreAllSameValue__(const T *idata, T *odata, int* oflag, const int n)
{
	__shared__ T   sdata[blockSize];
	__shared__ int sflag[blockSize];

	// each block of threads will work on blocksize^2 elements
	unsigned int work = blockSize * blockSize;

	unsigned int base = blockIdx.x * work;
	unsigned int tid = threadIdx.x;

	int my_flag = 1; //all same to start
	T   my_value = idata[base + tid];

	#pragma unroll
	for(int j=0; j<blockSize; j++)
	{
		const int k = base + j*blockSize + tid;
		if(k < n)
		{
			const T a = idata[k];

			my_flag &= equal<T>(a, my_value);
		}
	}

	sdata[tid] = my_value;
	sflag[tid] = my_flag;
	__syncthreads();

// #pragma unroll
	for(int j=2; j<=blockSize; j*=2)
	{
		if(tid < blockSize/j)// < n/(j-1))
		{
			sflag[tid] &= sflag[tid + blockSize/j];
		}
		__syncthreads();
	}

	if(tid == 0)
	{
		odata[blockIdx.x] = sdata[0];
		oflag[blockIdx.x] = sflag[0];
	}
}

#define BS 64
template <typename T>
bool arrayAreAllSameValue_(T* d_data, const int n, T& v)
{
	const int work = BS*BS;
	const int blocks = 1 + n / work;
	const int threads = BS;

	T* d_ws1;
	T* h_ws1;

	int* d_flag;
	int* h_flag;

	malloc_dh(&d_ws1,&h_ws1, sizeof(T)*n);
	malloc_dh(&d_flag,&h_flag, sizeof(int)*n);

	arrayAreAllSameValue__<BS, T><<<blocks, threads>>>(d_data, d_ws1, d_flag, n);
	KCHECK

	memcpy_d2h(h_ws1, d_ws1, sizeof(T)*blocks);
	memcpy_d2h(h_flag, d_flag, sizeof(int)*n);


	T res_val = h_ws1[0];
	int res_flag = h_flag[0];
	for(int i=1; i<blocks; i++)
	{
		res_flag &= h_flag[i];
	}

	free_dh(d_ws1, h_ws1);
	free_dh(d_flag, h_flag);

	v = res_val;

	return res_flag;
}

bool arrayAreAllSameValue(double* d_data, const int n, double& v)
{
	return arrayAreAllSameValue_<double>(d_data, n, v);
}
bool arrayAreAllSameValue(float* d_data, const int n, float& v)
{
	return arrayAreAllSameValue_<float>(d_data, n, v);
}
bool arrayAreAllSameValue(int* d_data, const int n, int& v)
{
	return arrayAreAllSameValue_<int>(d_data, n, v);
}
bool arrayAreAllSameValue(doubleComplex* d_data, const int n, doubleComplex& v)
{
	return arrayAreAllSameValue_<doubleComplex>(d_data, n, v);
}
bool arrayAreAllSameValue(floatComplex* d_data, const int n, floatComplex& v)
{
	return arrayAreAllSameValue_<floatComplex>(d_data, n, v);
}
