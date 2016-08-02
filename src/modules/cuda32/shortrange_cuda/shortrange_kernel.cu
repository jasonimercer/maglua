#include <complex>
#include <iostream>
#include <vector>
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>
#include <stdio.h>

#include "../core_cuda/spinsystem.hpp"

#define JM_FORWARD 1
#define JM_BACKWARD 0

#define REAL float
//#define CUCOMPLEX cuDoubleComplex
#define CUCOMPLEX cuFloatComplex
//#define MAKECOMPLEX(a,b) make_cuDoubleComplex(a,b) 
#define MAKECOMPLEX(a,b) make_cuFloatComplex(a,b) 

//#define SMART_SCHEDULE

#ifdef SMART_SCHEDULE
  #define IDX_PATT(a, b) \
	const int a = threadIdx.x; \
	const int b = blockIdx.x; 
#else
  #define IDX_PATT(a, b) \
	const int a = blockDim.x * blockIdx.x + threadIdx.x; \
	const int b = blockDim.y * blockIdx.y + threadIdx.y;
#endif


#if 1
#define BOUND_CHECKS 1
#define KCHECK \
{ \
	const cudaError_t i = cudaGetLastError();\
	if(i) \
		printf("(%s:%i) %s\n",  __FILE__, __LINE__-1, cudaGetErrorString(i));\
}
#define CHECKCALL(expression) \
{ \
	const cudaError_t err = (expression); \
	if(err != cudaSuccess) \
		printf("(%s:%i) (%i)%s\n", __FILE__, __LINE__, err, cudaGetErrorString(err)); \
	/* printf("(%s:%i) %s => %i\n", __FILE__, __LINE__, #expression, err); */ \
}

#else
#define KCHECK ;
#define CHECKERR(e) ;
#endif


// will operate on stacks of spins in the Z direction
__global__ void shortRangeABCollect(
			const int nx, const int ny, const int nz, 
			const float global_scale,
			float* h_A, const float* d_AB, int* d_offset, const float* d_sB,
			const int count, float sign)
{
	IDX_PATT(x, y);

	if(x >= nx || y >= ny)
		return;

	for(int ztarget=0; ztarget<nz; ztarget++)
	{
		const int zzz = ztarget * nx * ny;
		const int here = x + y*nx + zzz;
		for(unsigned int j=0; j<count; j++)
		{
			const float value = d_AB[j] * global_scale;
			const int dz = d_offset[j*3+2];
			const int other_spin_x = (x + d_offset[j*3+0]) % nx;
			const int other_spin_y = (y + d_offset[j*3+1]) % ny;
			const int other_spin_z = ztarget + dz;
			const int other = other_spin_x + nx*other_spin_y;
			if(other_spin_z >=0 && other_spin_z < nz)
			{
				if(dz < 0) //then need upper/lower neighbours
				{
					h_A[here] += sign * value * d_sB[other + other_spin_z*nx*ny];
				}
				else //covers case of dz=0 and dz>0
				{
					h_A[here] += value * d_sB[other + other_spin_z*nx*ny];
				}
			}
		}
	}
}



void JM_SHORTRANGE(const int nx, const int ny, const int nz, 
				   const float global_scale,
				   int* ABCount, int** d_ABoffset, float** d_ABvalue,
				   const float* sx, const float* sy, const float* sz,
				   float* hx, float* hy, float* hz)
{
	#ifdef SMART_SCHEDULE
	//different thread schedules for different access patterns
	dim3 blocks(nx);
	dim3 threads(ny);
	#else
	const int _blocksx = nx / 32 + 1;
	const int _blocksy = ny / 32 + 1;
	dim3 blocks(_blocksx, _blocksy);
	dim3 threads(32,32);
	#endif	


	ss_d_set3DArray32(hx, nx, ny, nz, 0);
	KCHECK;
	ss_d_set3DArray32(hy, nx, ny, nz, 0);
	KCHECK;
	ss_d_set3DArray32(hz, nx, ny, nz, 0);
	KCHECK;

#define XX 0
#define XY 1
#define XZ 2
#define YY 3
#define YZ 4
#define ZZ 5

	if(ABCount[XX])
	{
	    shortRangeABCollect<<<blocks, threads>>>(nx, ny, nz, global_scale, hx, d_ABvalue[XX], d_ABoffset[XX], sx, ABCount[XX],  1);
	    KCHECK;
	}
	if(ABCount[XY])
	{
	    shortRangeABCollect<<<blocks, threads>>>(nx, ny, nz, global_scale, hx, d_ABvalue[XY], d_ABoffset[XY], sy, ABCount[XY],  1);
	    KCHECK;
	}
	if(ABCount[XZ])
	{
	    shortRangeABCollect<<<blocks, threads>>>(nx, ny, nz, global_scale, hx, d_ABvalue[XZ], d_ABoffset[XZ], sz, ABCount[XZ], -1);
	    KCHECK;
	}


	if(ABCount[XY])
	{
	    shortRangeABCollect<<<blocks, threads>>>(nx, ny, nz, global_scale, hy, d_ABvalue[XY], d_ABoffset[XY], sx, ABCount[XY],  1);
	    KCHECK;
	}
	if(ABCount[YY])
	{
	    shortRangeABCollect<<<blocks, threads>>>(nx, ny, nz, global_scale, hy, d_ABvalue[YY], d_ABoffset[YY], sy, ABCount[YY],  1);
	    KCHECK;
	}
	if(ABCount[YZ])
	{
	    shortRangeABCollect<<<blocks, threads>>>(nx, ny, nz, global_scale, hy, d_ABvalue[YZ], d_ABoffset[YZ], sz, ABCount[YZ], -1);
	    KCHECK;
	}



	if(ABCount[XZ])
	{
	    shortRangeABCollect<<<blocks, threads>>>(nx, ny, nz, global_scale, hz, d_ABvalue[XZ], d_ABoffset[XZ], sx, ABCount[XZ], -1);
	    KCHECK;
	}
	if(ABCount[YZ])
	{
	    shortRangeABCollect<<<blocks, threads>>>(nx, ny, nz, global_scale, hz, d_ABvalue[YZ], d_ABoffset[YZ], sy, ABCount[YZ], -1);
	    KCHECK;
	}
	if(ABCount[ZZ])
	{
	    shortRangeABCollect<<<blocks, threads>>>(nx, ny, nz, global_scale, hz, d_ABvalue[ZZ], d_ABoffset[ZZ], sz, ABCount[ZZ],  1);
	    KCHECK;
	}
}
