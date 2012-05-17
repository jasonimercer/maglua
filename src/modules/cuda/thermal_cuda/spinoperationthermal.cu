#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define CHECK \
{ \
	const cudaError_t i = cudaGetLastError();\
	if(i) \
		printf("(%s:%i) %s\n",  __FILE__, __LINE__-1, cudaGetErrorString(i));\
}

#define IDX_PATT(a, b) \
	const int a = blockDim.x * blockIdx.x + threadIdx.x; \
	const int b = blockDim.y * blockIdx.y + threadIdx.y;
	
	
// FOOBAR = (2.0 * alpha * temperature) / (dt * gamma)
template <unsigned int twiddle>
__global__ void do_thermal(	
		const float* d_rng6,  
		double FOOBAR, double* d_scale,
		double* d_hx, double* d_hy, double* d_hz, double* d_ms,
		const int n)//x, const int ny, const int offset)
{
// 	IDX_PATT(x, y);
// 	if(x >= nx || y >= ny)
// 		return;
// 	const int idx = x + y*nx + offset;
	
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx >= n)
		return;
	
	const double ms = d_ms[idx];
	if(ms != 0)
	{
		const double stddev = sqrt((FOOBAR * d_scale[idx]) / ms);
		d_hx[idx] = stddev * d_rng6[idx*6+0+twiddle*3];
		d_hy[idx] = stddev * d_rng6[idx*6+1+twiddle*3];
		d_hz[idx] = stddev * d_rng6[idx*6+2+twiddle*3];
	}
	else
	{
		d_hx[idx] = 0;
		d_hy[idx] = 0;
		d_hz[idx] = 0;
	}
}


void cuda_thermal(const float* d_rng6, const int twiddle, 
	double alpha, double gamma, double dt, double temperature,
	double* d_hx, double* d_hy, double* d_hz, double* d_ms,
	double* d_scale,
	const int nxyz)//, const int ny, const int nz)
{
// 	const int nxyz = nx * ny * nz;
	const int threads = 512;
	const int blocks = nxyz / threads + 1;
		
	const double FOOBAR =  (2.0 * alpha * temperature) / (dt * gamma);
// 	const int _blocksx = nx / 32 + 1;
// 	const int _blocksy = ny / 32 + 1;
// 	dim3 blocks(_blocksx, _blocksy);
// 	dim3 threads(32,32);
	
	if(twiddle == 0)
	{
// 		for(int i=0; i<nz; i++)
// 		{
			do_thermal<0><<<blocks, threads>>>(d_rng6, FOOBAR, d_scale, d_hx, d_hy, d_hz, d_ms, nxyz);
			CHECK
// 		}
	}
	else
	{
// 		for(int i=0; i<nz; i++)
// 		{
			do_thermal<1><<<blocks, threads>>>(d_rng6, FOOBAR, d_scale, d_hx, d_hy, d_hz, d_ms, nxyz);
			CHECK
// 		}
	}
}
