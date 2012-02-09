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
	const double*     dmdt_tx, const double*     dmdt_ty, const double*     dmdt_tz,
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
		double MMh[3];
		double  Mh[3];

		// M x (H + ht) + M/|M| x (M x H)

		MH[0] = dmdt_sy[i] * dmdt_hz[i] - dmdt_sz[i] * dmdt_hy[i];
		MH[1] =(dmdt_sx[i] * dmdt_hz[i] - dmdt_sz[i] * dmdt_hx[i]) * -1.0;
		MH[2] = dmdt_sx[i] * dmdt_hy[i] - dmdt_sy[i] * dmdt_hx[i];
		
		Mh[0] = dmdt_sy[i] * (dmdt_hz[i] - dmdt_tz[i]) - dmdt_sz[i] * (dmdt_hy[i] - dmdt_ty[i]);
		Mh[1] =(dmdt_sx[i] * (dmdt_hz[i] - dmdt_tz[i]) - dmdt_sz[i] * (dmdt_hx[i] - dmdt_tx[i])) * -1.0;
		Mh[2] = dmdt_sx[i] * (dmdt_hy[i] - dmdt_ty[i]) - dmdt_sy[i] * (dmdt_hx[i] - dmdt_tx[i]);
		
		MMh[0]= dmdt_sy[i] * Mh[2] - dmdt_sz[i] * Mh[1];
		MMh[1]=(dmdt_sx[i] * Mh[2] - dmdt_sz[i] * Mh[0]) * -1.0;
		MMh[2]= dmdt_sx[i] * Mh[1] - dmdt_sy[i] * Mh[0];
		
		double gadt = gamma_dt / (1.0+alpha*alpha);
		double as = alpha/spinfrom_m[i];
		
		//reusing variables
		Mh[0] = spinfrom_sx[i] - gadt * (MH[0] + as * MMh[0]);
		Mh[1] = spinfrom_sy[i] - gadt * (MH[1] + as * MMh[1]);
		Mh[2] = spinfrom_sz[i] - gadt * (MH[2] + as * MMh[2]);
	      
		MMh[0] = 1.0 / sqrt(Mh[0]*Mh[0] + Mh[1]*Mh[1] + Mh[2]*Mh[2]);

		spinto_sx[i] = Mh[0] * spinfrom_m[i] * MMh[0];
		spinto_sy[i] = Mh[1] * spinfrom_m[i] * MMh[0];
		spinto_sz[i] = Mh[2] * spinfrom_m[i] * MMh[0];
	}
	
	
}
	
	
void cuda_llg_cart_apply(const int nx, const int ny, const int nz,
	double* dsx, double* dsy, double* dsz, double* dms, //dest (spinto)
	double* ssx, double* ssy, double* ssz, double* sms, // src (spinfrom)
	double* ddx, double* ddy, double* ddz, double* dds, // dm/dt spins
	double* htx, double* hty, double* htz,              // dm/dt thermal fields
	double* dhx, double* dhy, double* dhz,              // dm/dt fields
	const double alpha, const double dt, const double gamma)
{
	const int nxyz = nx*ny*nz;
	const int threads = 512;
	const int blocks = nxyz / threads + 1;
	
	llg_cart_apply<<<blocks, threads>>>(nxyz,
					ssx, ssy, ssz, sms,
					htx, hty, htz,
					dhx, dhy, dhz,
					ddx, ddy, ddz,
					dsx, dsy, dsz, dms,
					alpha, gamma*dt);
	CHECK
}

