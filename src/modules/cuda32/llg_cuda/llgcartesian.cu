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


__global__ void llg_cart_apply32(
	const int nxyz,
	const float* spinfrom_sx, const float* spinfrom_sy, const float* spinfrom_sz, const float* spinfrom_m,
	const float*     dmdt_tx, const float*     dmdt_ty, const float*     dmdt_tz,
	const float*     dmdt_hx, const float*     dmdt_hy, const float*     dmdt_hz,
	const float*     dmdt_sx, const float*     dmdt_sy, const float*     dmdt_sz,
	      float*   spinto_sx,       float*   spinto_sy,       float*   spinto_sz,       float* spinto_m,
	const float alpha, const float gamma_dt)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i >= nxyz)
		return;

	spinto_m[i] = spinfrom_m[i];
	
	if(spinto_m[i] > 0)
	{
		float  MH[3];
		float MMh[3];
		float  Mh[3];

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
		
		float gadt = gamma_dt / (1.0+alpha*alpha);
		float as = alpha/spinfrom_m[i];
		
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
	
				
void cuda_llg_cart_apply32(const int nx, const int ny, const int nz,
	float* dsx, float* dsy, float* dsz, float* dms, //dest (spinto)
	float* ssx, float* ssy, float* ssz, float* sms, // src (spinfrom)
	float* ddx, float* ddy, float* ddz, float* dds, // dm/dt spins
	float* htx, float* hty, float* htz,              // dm/dt thermal fields
	float* dhx, float* dhy, float* dhz,              // dm/dt fields
	const float alpha, const float dt, const float gamma)
{
	const int nxyz = nx*ny*nz;
	const int threads = 512;
	const int blocks = nxyz / threads + 1;
	
	llg_cart_apply32<<<blocks, threads>>>(nxyz,
					ssx, ssy, ssz, sms,
					htx, hty, htz,
					dhx, dhy, dhz,
					ddx, ddy, ddz,
					dsx, dsy, dsz, dms,
					alpha, gamma*dt);
	CHECK
}

