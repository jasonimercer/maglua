#include <cuda.h>
#include <cuda_runtime.h>
#include <math_functions.h>
#include <stdio.h>

#if __CUDA_ARCH__ >= 200
#define FAST_DIV(x,y) __ddiv_rn(x,y)
#else
#define FAST_DIV(x,y) ((x)/(y))
#endif

#define CHECK \
{ \
	const cudaError_t i = cudaGetLastError();\
	if(i) \
		printf("(%s:%i) %s\n",  __FILE__, __LINE__-1, cudaGetErrorString(i));\
}

__global__ void llg_cart_apply(
	const int nxyz,
	const double* spinfrom_sx, const double* spinfrom_sy, const double* spinfrom_sz, const double* spinfrom_m,
	const double*     dmdt_hx, const double*     dmdt_hy, const double*     dmdt_hz,
	const double*     dmdt_sx, const double*     dmdt_sy, const double*     dmdt_sz,
	      double*   spinto_sx,       double*   spinto_sy,       double*   spinto_sz,       double* spinto_m,
	const double alpha, const double gamma_dt)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i >= nxyz)
		return;

	spinto_m[i] = spinfrom_m[i];
	
	if(spinto_m[i] > 0)
	{
		double  MH[3];
		double MMH[3];
		MH[0] = dmdt_sy[i] * dmdt_hz[i] - dmdt_sz[i] * dmdt_hy[i];
		MH[1] =(dmdt_sx[i] * dmdt_hz[i] - dmdt_sz[i] * dmdt_hx[i]) * -1.0;
		MH[2] = dmdt_sx[i] * dmdt_hy[i] - dmdt_sy[i] * dmdt_hx[i];
		
		MMH[0]= dmdt_sy[i] * MH[2] - dmdt_sz[i] * MH[1];
		MMH[1]=(dmdt_sx[i] * MH[2] - dmdt_sz[i] * MH[0]) * -1.0;
		MMH[2]= dmdt_sx[i] * MH[1] - dmdt_sy[i] * MH[0];
		
		double gadt = gamma_dt / (1.0+alpha*alpha);
		double as = alpha/spinfrom_m[i];
		
		//reusing variables
		MH[0] = spinfrom_sx[i] - gadt * (MH[0] + as * MMH[0]);
		MH[1] = spinfrom_sy[i] - gadt * (MH[1] + as * MMH[1]);
		MH[2] = spinfrom_sz[i] - gadt * (MH[2] + as * MMH[2]);

		
		MMH[0] = 1.0 / sqrt(MH[0]*MH[0] + MH[1]*MH[1] + MH[2]*MH[2]);

		spinto_sx[i] = MH[0] * spinfrom_m[i] * MMH[0];
		spinto_sy[i] = MH[1] * spinfrom_m[i] * MMH[0];
		spinto_sz[i] = MH[2] * spinfrom_m[i] * MMH[0];
	}
	
	
}
	
	
void cuda_llg_cart_apply(const int nx, const int ny, const int nz,
	double* dsx, double* dsy, double* dsz, double* dms, //dest (spinto)
	double* ssx, double* ssy, double* ssz, double* sms, // src (spinfrom)
	double* ddx, double* ddy, double* ddz, double* dds, // dm/dt spins
	double* dhx, double* dhy, double* dhz,              // dm/dt fields
	const double alpha, const double dt, const double gamma)
{
	const int nxyz = nx*ny*nz;
	const int threads = 512;
	const int blocks = nxyz / threads + 1;
	
	llg_cart_apply<<<blocks, threads>>>(nxyz,
					ssx, ssy, ssz, sms,
					dhx, dhy, dhz,
					ddx, ddy, ddz,
					dsx, dsy, dsz, dms,
					alpha, gamma*dt);
	CHECK
}

