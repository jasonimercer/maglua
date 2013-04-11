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
	
template<int do_renormalize, int thermal_both_terms>
__global__ void llg_cart_apply_N(const int nxyz, 
	double** spinto_x, double** spinto_y, double** spinto_z, double** spinto_m,
	double** spinfrom_x, double** spinfrom_y, double** spinfrom_z, double** spinfrom_m,
	double** dmdt_sx,    double** dmdt_sy,    double** dmdt_sz, /* double** dmdt_sm, */
	double** dmdt_tx, double** dmdt_ty, double** dmdt_tz, //thermal
	double** dmdt_hx, double** dmdt_hy, double** dmdt_hz, //sum (all)
	double* dt, double** d_alpha_N, double* d_alpha, double** d_gamma_N, double* d_gamma,
	const int n)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i >= nxyz) return;
	
	const int j = blockDim.y * blockIdx.y + threadIdx.y;
	if(j >= n) return;
	
	if(dmdt_hx[j] == 0 || dmdt_hy[j] == 0 || dmdt_hz[j] == 0)
		return;

	double alpha;
	double gamma;
	
	if(d_alpha_N[j])
	{
		alpha = d_alpha_N[j][i];
	}
	else
	{
		alpha =  d_alpha[j];
	}
	
	if(d_gamma_N[j])
	{
		gamma = d_gamma_N[j][i];
	}
	else
	{
		gamma =  d_gamma[j];
	}

	
	const double gamma_dt = gamma * dt[j];
	
	
	spinto_m[j][i] = spinfrom_m[j][i];
	
	if(spinto_m[j][i] > 0)
	{
		double  MH[3];
		double  FirstTerm[3];
		double  SecondTerm[3];
	
		if(thermal_both_terms == 0) //so thermal only in 1st term, subtracting out of 2nd term
		{
			double h[3];
			if(dmdt_tx[j])
			{
				h[0] = dmdt_hx[j][i] - dmdt_tx[j][i];
				h[1] = dmdt_hy[j][i] - dmdt_ty[j][i];
				h[2] = dmdt_hz[j][i] - dmdt_tz[j][i];
			}
			else
			{
				h[0] = dmdt_hx[j][i];
				h[1] = dmdt_hy[j][i];
				h[2] = dmdt_hz[j][i];
			}
			
			CROSS(MH, 	dmdt_sx[j][i], dmdt_sy[j][i], dmdt_sz[j][i],  h[0], h[1], h[2]); // really Mh
			CROSS(SecondTerm,  dmdt_sx[j][i], dmdt_sy[j][i], dmdt_sz[j][i],  MH[0], MH[1], MH[2]); // MMh
			
			CROSS(FirstTerm, 	dmdt_sx[j][i], dmdt_sy[j][i], dmdt_sz[j][i], dmdt_hx[j][i], dmdt_hy[j][i], dmdt_hz[j][i]); // MH
		}
		if(thermal_both_terms == 1) // thermal in both, no need to subtract
		{
			// M x (H ) + M/|M| x (M x H)
			CROSS(FirstTerm, dmdt_sx[j][i], dmdt_sy[j][i], dmdt_sz[j][i],   dmdt_hx[j][i], dmdt_hy[j][i], dmdt_hz[j][i]);

			CROSS(SecondTerm, dmdt_sx[j][i], dmdt_sy[j][i], dmdt_sz[j][i],   FirstTerm[0], FirstTerm[1], FirstTerm[2]);
		}
		
		const double gadt = gamma_dt / (1.0+alpha*alpha);
		const double as = alpha/spinfrom_m[j][i];
		
		//reusing variables
		MH[0] = spinfrom_x[j][i] - gadt * (FirstTerm[0] + as * SecondTerm[0]);
		MH[1] = spinfrom_y[j][i] - gadt * (FirstTerm[1] + as * SecondTerm[1]);
		MH[2] = spinfrom_z[j][i] - gadt * (FirstTerm[2] + as * SecondTerm[2]);
		
		if(do_renormalize == 1)
		{
			//renormalize step, reusing variable
			FirstTerm[0] = 1.0 / sqrt(MH[0]*MH[0] + MH[1]*MH[1] + MH[2]*MH[2]);

			spinto_x[j][i] = MH[0] * spinfrom_m[j][i] * FirstTerm[0];
			spinto_y[j][i] = MH[1] * spinfrom_m[j][i] * FirstTerm[0];
			spinto_z[j][i] = MH[2] * spinfrom_m[j][i] * FirstTerm[0];
		}
		else
		{
			spinto_x[j][i] = MH[0];
			spinto_y[j][i] = MH[1];
			spinto_z[j][i] = MH[2];
		}
	}
}


				
void cuda_llg_cart_apply_N(	int nx, int ny, int nz,
	double** dsx, double** dsy, double** dsz, double** dms, //dest (spinto)
	double** ssx, double** ssy, double** ssz, double** sms, // src (spinfrom)
	double** ddx, double** ddy, double** ddz, double** dds, // dm/dt spins
	double** htx, double** hty, double** htz,              // dm/dt thermal fields
	double** dhx, double** dhy, double** dhz,              // dm/dt fields
	double* dt, double** d_alpha_N, double* d_alpha, double** d_gamma_N, double* d_gamma,
	int thermalOnlyFirstTerm, int disableRenormalization, const int n
)
{
	const int nxyz = nx*ny*nz;
	const int threadsX = 512;
	const int blocksX = nxyz / threadsX + 1;

	const int threadsY = 1;
	const int blocksY = n;
	
	dim3 gd(blocksX, blocksY);
	dim3 bd(threadsX, threadsY);
	
	if(thermalOnlyFirstTerm)
	{
		if(disableRenormalization)
			llg_cart_apply_N<0, 0><<<gd, bd>>>(nxyz,
							dsx, dsy, dsz, dms,
							ssx, ssy, ssz, sms,
							ddx, ddy, ddz,
							htx, hty, htz,
							dhx, dhy, dhz,
							dt, d_alpha_N, d_alpha, d_gamma_N, d_gamma, n);
		else
			llg_cart_apply_N<1, 0><<<gd, bd>>>(nxyz,
							dsx, dsy, dsz, dms,
							ssx, ssy, ssz, sms,
							ddx, ddy, ddz,
							htx, hty, htz,
							dhx, dhy, dhz,
							dt, d_alpha_N, d_alpha, d_gamma_N, d_gamma, n);
			
	}
	else
	{
		if(disableRenormalization)
			llg_cart_apply_N<0, 1><<<gd, bd>>>(nxyz,
							dsx, dsy, dsz, dms,
							ssx, ssy, ssz, sms,
							ddx, ddy, ddz,
							htx, hty, htz,
							dhx, dhy, dhz,
							dt, d_alpha_N, d_alpha, d_gamma_N, d_gamma, n);
		else
			llg_cart_apply_N<1, 1><<<gd, bd>>>(nxyz,
							dsx, dsy, dsz, dms,
							ssx, ssy, ssz, sms,
							ddx, ddy, ddz,
							htx, hty, htz,
							dhx, dhy, dhz,
							dt, d_alpha_N, d_alpha, d_gamma_N, d_gamma, n);
			
	}
	CHECK
}
