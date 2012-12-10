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

#define CROSS(v, a0, a1, a2, b0, b1, b2) \
	v[0] = (a1) * (b2) - (a2) * (b1); \
	v[1] = (a2) * (b0) - (a0) * (b2); \
	v[2] = (a0) * (b1) - (a1) * (b0);
	
__global__ void llg_cart_apply(
	const int nxyz,
	const double* spinfrom_sx, const double* spinfrom_sy, const double* spinfrom_sz, const double* spinfrom_m,
	const double*     dmdt_tx, const double*     dmdt_ty, const double*     dmdt_tz,
	const double*     dmdt_hx, const double*     dmdt_hy, const double*     dmdt_hz,
	const double*     dmdt_sx, const double*     dmdt_sy, const double*     dmdt_sz,
	      double*   spinto_sx,       double*   spinto_sy,       double*   spinto_sz,       double* spinto_m,
	const double dt, const double _alpha, const double* d_alpha, const double _gamma, const double* d_gamma)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i >= nxyz)
		return;

	double alpha;
	double gamma;
	
	if(d_alpha)
	{
		alpha = d_alpha[i];
	}
	else
	{
		alpha =  _alpha;
	}

	if(d_gamma)
	{
		gamma = d_gamma[i];
	}
	else
	{
		gamma = _gamma;
	}
	
	const double gamma_dt = gamma * dt;
	
	
	spinto_m[i] = spinfrom_m[i];
	
	if(spinto_m[i] > 0)
	{
		double  MHh[3];
		double MMH[3];
		double  MH[3];

		// M x (H + ht) + M/|M| x (M x H)

		CROSS(MHh, dmdt_sx[i], dmdt_sy[i], dmdt_sz[i],   dmdt_hx[i], dmdt_hy[i], dmdt_hz[i]);

		CROSS(MH, 	dmdt_sx[i], dmdt_sy[i], dmdt_sz[i],   
					dmdt_hx[i] - dmdt_tx[i], 
					dmdt_hy[i] - dmdt_ty[i], 
					dmdt_hz[i] - dmdt_tz[i]);
		
		CROSS(MMH, dmdt_sx[i], dmdt_sy[i], dmdt_sz[i],   MH[0], MH[1], MH[2]);
		
		const double gadt = gamma_dt / (1.0+alpha*alpha);
		const double as = alpha/spinfrom_m[i];
		
		//reusing variables
		MH[0] = spinfrom_sx[i] - gadt * (MHh[0] + as * MMH[0]);
		MH[1] = spinfrom_sy[i] - gadt * (MHh[1] + as * MMH[1]);
		MH[2] = spinfrom_sz[i] - gadt * (MHh[2] + as * MMH[2]);
	      
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
	double* htx, double* hty, double* htz,              // dm/dt thermal fields
	double* dhx, double* dhy, double* dhz,              // dm/dt fields
	const double dt, const double alpha, const double* d_alpha, const double gamma, const double* d_gamma)
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
					dt, alpha, d_alpha, gamma, d_gamma);
	CHECK
}

