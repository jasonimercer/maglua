#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>

#include "spinoperationanisotropy.hpp"


__global__ void do_anisotropy_N(const double global_scale,
	const double** d_sx_N, const double** d_sy_N, const double** d_sz_N,
	const double* d_nx, const double* d_ny, const double* d_nz, const double* d_k,
	double** d_hx_N, double** d_hy_N, double** d_hz_N, 
	const int nxyz, const int n)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= nxyz) return;
	
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if(j >= n) return;
	
	const double* d_sx = d_sx_N[j];
	const double* d_sy = d_sy_N[j];
	const double* d_sz = d_sz_N[j];

	double* d_hx = d_hx_N[j];
	double* d_hy = d_hy_N[j];
	double* d_hz = d_hz_N[j];
	
	const double ms2 = d_sx[i]*d_sx[i] + d_sy[i]*d_sy[i] + d_sz[i]*d_sz[i];
	
	if(ms2 > 0)
	{
		const double SpinDotEasyAxis = 
			d_sx[i]*d_nx[i] + d_sy[i]*d_ny[i] + d_sz[i]*d_nz[i];
	
		const double v = global_scale * 2.0 * d_k[i] * SpinDotEasyAxis / ms2;
		
		d_hx[i] = d_nx[i] * v;
		d_hy[i] = d_ny[i] * v;
		d_hz[i] = d_nz[i] * v;
	}
	else
	{
		d_hx[i] = 0;
		d_hy[i] = 0;
		d_hz[i] = 0;
	}
}

void cuda_anisotropy_N(const double global_scale,
	const double** d_sx, const double** d_sy, const double** d_sz,
	const double* d_nx, const double* d_ny, const double* d_nz, const double* d_k,
	double** d_hx, double** d_hy, double** d_hz,
	const int nxyz, const int n)
{
	const int threadsX = 256;
	const int blocksX = nxyz / threadsX + 1;

	const int threadsY = 1;
	const int blocksY = n;
	
	dim3 gd(blocksX, blocksY);
	dim3 bd(threadsX, threadsY);
	
	
// 	const int blocksx = nx / 32 + 1;
// 	const int blocksy = ny / 32 + 1;

// 	dim3 blocks(blocksx, blocksy);
// 	dim3 threads(32, 32);

// 	for(int z=0; z<nz; z++)
// 	{
// 		const int offset = z * nx * ny;
		do_anisotropy_N<<<gd, bd>>>(
				global_scale,
				d_sx, d_sy, d_sz,
				d_nx, d_ny, d_nz, d_k,
				d_hx, d_hy, d_hz, 
				nxyz, n);
// 	}
}





__global__ void do_anisotropy_compressed_N(const double global_scale,
	const double** d_sx_N, const double** d_sy_N, const double** d_sz_N,
	const double* d_LUT, const char* d_idx,
	double** d_hx_N, double** d_hy_N, double** d_hz_N, 
	const int nxyz, int n)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= nxyz) return;
	
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if(j >= n) return;
	
	const double* d_sx = d_sx_N[j];
	const double* d_sy = d_sy_N[j];
	const double* d_sz = d_sz_N[j];

	double* d_hx = d_hx_N[j];
	double* d_hy = d_hy_N[j];
	double* d_hz = d_hz_N[j];
	
	
	
	const double nx = d_LUT[d_idx[i]*4+0];
	const double ny = d_LUT[d_idx[i]*4+1];
	const double nz = d_LUT[d_idx[i]*4+2];
	const double k  = d_LUT[d_idx[i]*4+3];
	
	const double ms2 = d_sx[i]*d_sx[i] + d_sy[i]*d_sy[i] + d_sz[i]*d_sz[i];
	
	if(ms2 > 0)
	{
		const double SpinDotEasyAxis = 
			d_sx[i]*nx + d_sy[i]*ny + d_sz[i]*nz;
	
		const double v = global_scale * 2.0 * k * SpinDotEasyAxis / ms2;
		
		d_hx[i] = nx * v;
		d_hy[i] = ny * v;
		d_hz[i] = nz * v;
	}
	else
	{
		d_hx[i] = 0;
		d_hy[i] = 0;
		d_hz[i] = 0;
	}
}


void cuda_anisotropy_compressed_N(const double global_scale,
	const double** d_sx, const double** d_sy, const double** d_sz,
	const double* d_LUT, const char* d_idx,
	double** d_hx, double** d_hy, double** d_hz,
	const int nxyz, const int n)
{
	const int threadsX = 256;
	const int blocksX = nxyz / threadsX + 1;

	const int threadsY = 1;
	const int blocksY = n;
	
	dim3 gd(blocksX, blocksY);
	dim3 bd(threadsX, threadsY);
	
	
// 	const int blocksx = nx / 32 + 1;
// 	const int blocksy = ny / 32 + 1;

// 	dim3 blocks(blocksx, blocksy);
// 	dim3 threads(32, 32);

// 	for(int z=0; z<nz; z++)
// 	{
// 		const int offset = z * nx * ny;
		do_anisotropy_compressed_N<<<gd, bd>>>(
				global_scale,
				d_sx, d_sy, d_sz,
				d_LUT, d_idx,
				d_hx, d_hy, d_hz, 
				nxyz, n);
		
		
// 	const int blocks = nxyz / 1024 + 1;
// 	const int threads = 1024;
// 
// 	do_anisotropy_compressed<<<blocks, threads>>>(
// 		    global_scale,
// 			d_sx, d_sy, d_sz,
// 			d_LUT, d_idx,
// 			d_hx, d_hy, d_hz, 
// 			nxyz);
}
