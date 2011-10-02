#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>
#include <complex>
#include <stdio.h>
using namespace std;

// #include "./mtimer.h"

#define JM_FORWARD 1
#define JM_BACKWARD 0

#define REAL double
#define CUCOMPLEX cuDoubleComplex
#define MAKECOMPLEX(a,b) make_cuDoubleComplex(a,b) 

#define CHECK \
{ \
	const cudaError_t i = cudaGetLastError();\
	if(i) \
		printf("(%s:%i) %s\n",  __FILE__, __LINE__-1, cudaGetErrorString(i));\
}

#define BIT(patt, num) ((patt & (1 << num)) >> num)
static int bit_rev_perm(unsigned int i, unsigned int log2n)
{
	int k, p;
	int j = 0;
	for(k=0, p=log2n-1; k<log2n; k++, p--)
	{
		j |= BIT(i, k) << p;
	}
	return j;
}

#define PI_II complex<double>(0,3.14159265358979)
static complex<double> get_exp(const int step, const int i, int dir)
{
	unsigned int bits = (2 << step) - 1;
	unsigned int half = (1 << step);
	const unsigned int hbits = half - 1;

	complex<double> p = -2.0*PI_II / ((double)(2<<(step)));
	
	if(dir == JM_BACKWARD)
	{
		p *= -1.0;
	}
	
	const int j  = i & bits; // j = count in subset
	const int q  = i &hbits; // count in half set
	const int hi = j & (1 << step);

	if(hi)
		return -1.0 * exp(p * double(q));
	return exp(p * double(q));
}


__global__ void Fi_2D_x(const int N_x, const int N_y, const int step,  CUCOMPLEX* dst,  CUCOMPLEX* src, CUCOMPLEX* exp2pi)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(i >= N_x || y >= N_y)
		return;
	
	const unsigned int  bits = (2 << step) - 1;
	const unsigned int  half = (1 << step);
	const unsigned int hbits = half - 1;
	const unsigned int base  = (0xFFFF ^ bits);

	const int j   = i &  bits; // count in subset
	const int q   = i & hbits; // count in half set
	const int k   = i &  base; // base

	//dst[k+j] = src[k+q] + exp2pi[step*N+i] * src[k+q+half];
	dst[k+j + y*N_x] = cuCadd(src[k+q + y*N_x], cuCmul(exp2pi[step*N_x+i], src[k+q+half + y*N_x]));
}


__global__ void Fi_2D_y(const int N_x, const int N_y, const int step,  CUCOMPLEX* dst,  CUCOMPLEX* src, CUCOMPLEX* exp2pi)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(i >= N_x || y >= N_y)
		return;
	
	const unsigned int  bits = (2 << step) - 1;
	const unsigned int  half = (1 << step);
	const unsigned int hbits = half - 1;
	const unsigned int base  = (0xFFFF ^ bits);

	const int j   = y &  bits; // count in subset
	const int q   = y & hbits; // count in half set
	const int k   = y &  base; // base

	//dst[k+j] = src[k+q] + exp2pi[step*N+i] * src[k+q+half];
	dst[(k+j)*N_x + i] = cuCadd(src[(k+q)*N_x + i], cuCmul(exp2pi[step*N_y+y], src[(k+q+half)*N_x + i]));
}

__global__ void twiddle2D(const int N_x, const int N_y, int* d_brp_x, int* d_brp_y, CUCOMPLEX* d_dest,  REAL* d_src)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(i >= N_x || y >= N_y)
		return;

	const int ti = d_brp_x[i];
	const int ty = d_brp_y[y];
	
	d_dest[ti + ty * N_x] = MAKECOMPLEX(d_src[i + y * N_x], 0);
}
__global__ void twiddle2DCC(const int N_x, const int N_y, int* d_brp_x, int* d_brp_y, CUCOMPLEX* d_dest,  CUCOMPLEX* d_src)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(i >= N_x || y >= N_y)
		return;

	const int ti = d_brp_x[i];
	const int ty = d_brp_y[y];
	
	d_dest[ti + ty * N_x] = d_src[i + y * N_x];
}

__global__ void scaleC(const int N_x, const int N_y, CUCOMPLEX* d, CUCOMPLEX* s,  double v)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(i >= N_x || y >= N_y)
		return;
	
	const int idx = i + y * N_x;
	d[idx].x = v * s[idx].x;
	d[idx].y = v * s[idx].y;
}
__global__ void setC(const int N_x, const int N_y, CUCOMPLEX* v,  double R, double I)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(i >= N_x || y >= N_y)
		return;
	
	const int idx = i + y * N_x;
	v[idx].x = R;
	v[idx].y = I;
}


typedef struct JM_LONGRANGE_PLAN
{
	int N_x, N_y, N_z;
	int log2N_x, log2N_y;
	
	// bit reversal permutation
	int* d_brp_x;
	int* d_brp_y;
	
	CUCOMPLEX* d_exp2pi_x_f; //this will hold exp factors
	CUCOMPLEX* d_exp2pi_y_f;
	
	CUCOMPLEX* d_exp2pi_x_b; //backward
	CUCOMPLEX* d_exp2pi_y_b;
	
	REAL* d_input;
	REAL* d_output;
	
	REAL* h_input; //write-only memory for inputting r2c 
	REAL* h_output; // field "output"
	
	// 2D arrays, 1st dimmension is for layer
	CUCOMPLEX** d_sx_q;
	CUCOMPLEX** d_sy_q;
	CUCOMPLEX** d_sz_q;
	
	CUCOMPLEX** d_hx_q;
	CUCOMPLEX** d_hy_q;
	CUCOMPLEX** d_hz_q;
	
	// 2D arrays, 1st dimmension is for interlayer offset
	CUCOMPLEX** d_GammaXX;
	CUCOMPLEX** d_GammaXY;
	CUCOMPLEX** d_GammaXZ;
	
	CUCOMPLEX** d_GammaYY;
	CUCOMPLEX** d_GammaYZ;
	
	CUCOMPLEX** d_GammaZZ;
	
	CUCOMPLEX* d_A; //chunks of memory for out-of-place FFT calcs
	CUCOMPLEX* d_B;
}JM_LONGRANGE_PLAN;

JM_LONGRANGE_PLAN* make_JM_LONGRANGE_PLAN(int N_x, int N_y, int N_z, 
	double* GammaXX, double* GammaXY, double* GammaXZ,
	                 double* GammaYY, double* GammaYZ,
	                                  double* GammaZZ)
{
	const int nz = N_z + N_z - 1;
	const int log2N_x = log2((double)N_x);
	const int log2N_y = log2((double)N_y);
	const int sRxy = sizeof(REAL) * N_x * N_y;
	const int sCxy = sizeof(CUCOMPLEX) * N_x * N_y;
	
	JM_LONGRANGE_PLAN* p = new JM_LONGRANGE_PLAN;
	p->N_x = N_x;
	p->N_y = N_y;
	p->N_z = N_z;
	p->log2N_x = log2N_x;
	p->log2N_y = log2N_y;
	
	// Building what is needed for future FFTs
	int* h_brp_x;
	int* h_brp_y;
	cudaHostAlloc(&h_brp_x, sizeof(int) * N_x, cudaHostAllocWriteCombined); //host "write-only"
	cudaHostAlloc(&h_brp_y, sizeof(int) * N_y, cudaHostAllocWriteCombined); //host "write-only"
	cudaMalloc(&(p->d_brp_x), sizeof(int) * N_x);
	cudaMalloc(&(p->d_brp_y), sizeof(int) * N_y);

	for(int i=0; i<N_x; i++)
		h_brp_x[i] = bit_rev_perm(i, log2N_x);
	for(int i=0; i<N_y; i++)
		h_brp_y[i] = bit_rev_perm(i, log2N_y);
	
	cudaMemcpy(p->d_brp_x, h_brp_x, sizeof(int) * N_x, cudaMemcpyHostToDevice);
	cudaMemcpy(p->d_brp_y, h_brp_y, sizeof(int) * N_y, cudaMemcpyHostToDevice);
	cudaFreeHost(h_brp_x);
	cudaFreeHost(h_brp_y);

	CUCOMPLEX* h_exp2pi_x;
	CUCOMPLEX* h_exp2pi_y; 
	cudaHostAlloc(&h_exp2pi_x, sizeof(CUCOMPLEX) * N_x * log2N_x, cudaHostAllocWriteCombined); //host "write-only"
	cudaHostAlloc(&h_exp2pi_y, sizeof(CUCOMPLEX) * N_y * log2N_y, cudaHostAllocWriteCombined); //host "write-only"
	cudaMalloc(&(p->d_exp2pi_x_f), sizeof(CUCOMPLEX) * N_x * log2N_x);
	cudaMalloc(&(p->d_exp2pi_y_f), sizeof(CUCOMPLEX) * N_y * log2N_y);
	cudaMalloc(&(p->d_exp2pi_x_b), sizeof(CUCOMPLEX) * N_x * log2N_x);
	cudaMalloc(&(p->d_exp2pi_y_b), sizeof(CUCOMPLEX) * N_y * log2N_y);

	for(int j=0; j<log2N_x; j++)
	{
		for(int i=0; i<N_x; i++)
		{
			complex<double> x = get_exp(j, i, JM_FORWARD);
			h_exp2pi_x[i + j*N_x] = MAKECOMPLEX(x.real(), x.imag());
		}
	}
	for(int j=0; j<log2N_y; j++)
	{
		for(int i=0; i<N_y; i++)
		{
			complex<double> x = get_exp(j, i, JM_FORWARD);
			h_exp2pi_y[i + j*N_y] = MAKECOMPLEX(x.real(), x.imag());
		}
	}
	
	cudaMemcpy(p->d_exp2pi_x_f, h_exp2pi_x, sizeof(CUCOMPLEX) * N_x * log2N_x, cudaMemcpyHostToDevice);
	cudaMemcpy(p->d_exp2pi_y_f, h_exp2pi_y, sizeof(CUCOMPLEX) * N_y * log2N_y, cudaMemcpyHostToDevice);

	for(int j=0; j<log2N_x; j++)
	{
		for(int i=0; i<N_x; i++)
		{
			complex<double> x = get_exp(j, i, JM_BACKWARD);
			h_exp2pi_x[i + j*N_x] = MAKECOMPLEX(x.real(), x.imag());
		}
	}
	for(int j=0; j<log2N_y; j++)
	{
		for(int i=0; i<N_y; i++)
		{
			complex<double> x = get_exp(j, i, JM_BACKWARD);
			h_exp2pi_y[i + j*N_y] = MAKECOMPLEX(x.real(), x.imag());
		}
	}
	
	cudaMemcpy(p->d_exp2pi_x_b, h_exp2pi_x, sizeof(CUCOMPLEX) * N_x * log2N_x, cudaMemcpyHostToDevice);
	cudaMemcpy(p->d_exp2pi_y_b, h_exp2pi_y, sizeof(CUCOMPLEX) * N_y * log2N_y, cudaMemcpyHostToDevice);
	
	cudaFreeHost(h_exp2pi_x);
	cudaFreeHost(h_exp2pi_y);
	// DONE making parts needed for future FFTs
	
	cudaMalloc(&(p->d_A), sCxy);
	cudaMalloc(&(p->d_B), sCxy);
	

	// 2D arrays, 1st dimmension is for layer
	p->d_sx_q = new CUCOMPLEX*[nz];
	p->d_sy_q = new CUCOMPLEX*[nz];
	p->d_sz_q = new CUCOMPLEX*[nz];

	p->d_hx_q = new CUCOMPLEX*[nz];
	p->d_hy_q = new CUCOMPLEX*[nz];
	p->d_hz_q = new CUCOMPLEX*[nz];

	for(int i=0; i<nz; i++)
	{
		cudaMalloc(&(p->d_sx_q[i]), sCxy);
		cudaMalloc(&(p->d_sy_q[i]), sCxy);
		cudaMalloc(&(p->d_sz_q[i]), sCxy);
	
		cudaMalloc(&(p->d_hx_q[i]), sCxy);
		cudaMalloc(&(p->d_hy_q[i]), sCxy);
		cudaMalloc(&(p->d_hz_q[i]), sCxy);
	}
	
	// make room for FT'd interaction matrices
	p->d_GammaXX = new CUCOMPLEX*[nz];
	p->d_GammaXY = new CUCOMPLEX*[nz];
	p->d_GammaXZ = new CUCOMPLEX*[nz];

	p->d_GammaYY = new CUCOMPLEX*[nz];
	p->d_GammaYZ = new CUCOMPLEX*[nz];
	
	p->d_GammaZZ = new CUCOMPLEX*[nz];

	for(int i=0; i<nz; i++)
	{
		cudaMalloc(&(p->d_GammaXX[i]), sCxy);
		cudaMalloc(&(p->d_GammaXY[i]), sCxy);
		cudaMalloc(&(p->d_GammaXZ[i]), sCxy);
	
		cudaMalloc(&(p->d_GammaYY[i]), sCxy);
		cudaMalloc(&(p->d_GammaYZ[i]), sCxy);
	
		cudaMalloc(&(p->d_GammaZZ[i]), sCxy);
	}
	
	
	cudaHostAlloc(&(p->h_input), sRxy, cudaHostAllocWriteCombined); //host "write-only"
	cudaHostAlloc(&(p->h_output),sRxy, 0);

	cudaMalloc(&(p->d_input), sRxy);
	cudaMalloc(&(p->d_output),sRxy);
	
	// now we will work on loading all the interaction matrices
	// onto the GPU and fourier transforming them
	static const struct {
        double* h; //host memory
		CUCOMPLEX** d; //device memory
    } sd[] = { //sd = static data
        {GammaXX, p->d_GammaXX},
        {GammaXY, p->d_GammaXY},
        {GammaXZ, p->d_GammaXZ},
        {GammaYY, p->d_GammaYY},
        {GammaYZ, p->d_GammaYZ},
        {GammaZZ, p->d_GammaZZ},
		{0,0}
	};

	
	const int blocksx = N_x / 32 + 1;
	const int blocksy = N_y / 32 + 1;
	
	dim3 blocks(blocksx, blocksy);
	dim3 threads(32, 32);
	
	CUCOMPLEX* d_src  = p->d_A; //local vars for swapping workspace
	CUCOMPLEX* d_dest = p->d_B;
	CUCOMPLEX* d_tmp = 0;
	
	for(int k=0; sd[k].h; k++)
	{
		for(int j=0; j<nz; j++)
		{
			d_src = p->d_A;
			d_dest = p->d_B;
			
			memcpy(p->h_input, &(sd[k].h[j*N_x*N_y]), sRxy);
			cudaMemcpy(p->d_input, p->h_input,sRxy, cudaMemcpyHostToDevice);

			// twiddle the data, next step will be to fft it. 
			twiddle2D<<<blocks, threads>>>(N_x, N_y, p->d_brp_x, p->d_brp_y, p->d_A, p->d_input);
			CHECK
			
			for(int i=0; i<log2N_x; i++)
			{
				Fi_2D_x<<<blocks, threads>>>(N_x, N_y, i, d_dest, d_src, p->d_exp2pi_x_f);
				CHECK
			
				d_tmp = d_src;
				d_src = d_dest;
				d_dest = d_tmp;
			}
			for(int i=0; i<log2N_y; i++)
			{
				Fi_2D_y<<<blocks, threads>>>(N_x, N_y, i, d_dest, d_src, p->d_exp2pi_y_f);
				CHECK

				d_tmp = d_src;
				d_src = d_dest;
				d_dest = d_tmp;
			}
			
			// going to prescale the data into d_GammaAB:
			scaleC<<<blocks, threads>>>(N_x, N_y, sd[k].d[j], d_src, 1.0/((double)(N_x * N_y)));
			CHECK
		}
	}



	
	return p;
	
}

void free_JM_LONGRANGE_PLAN(JM_LONGRANGE_PLAN* p)
{
#warning need to free plan
}


__global__ void convolve(const int N_x, const int N_y, CUCOMPLEX* d_dest,  CUCOMPLEX* d_A,  CUCOMPLEX* d_B)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(i >= N_x || y >= N_y)
		return;

	d_dest[i + y * N_x] = cuCmul(d_A[i + y * N_x], d_B[i + y * N_x]);
}

__global__ void convolveSum(const int N_x, const int N_y, CUCOMPLEX* d_dest,  CUCOMPLEX* d_A,  CUCOMPLEX* d_B)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(i >= N_x || y >= N_y)
		return;

	d_dest[i + y * N_x] = cuCadd(d_dest[i + y * N_x], cuCmul(d_A[i + y * N_x], d_B[i + y * N_x]));
}

__global__ void getRPart(const int N_x, const int N_y, REAL* d_dest,  CUCOMPLEX* d_src)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	
	if(i >= N_x || y >= N_y)
		return;

	d_dest[i + y * N_x] = d_src[i + y * N_x].x;
}

void JM_LONGRANGE(JM_LONGRANGE_PLAN* p, 
				  double* sx, double* sy, double* sz,
				  double* hx, double* hy, double* hz)
{
// 	timer = new_timer(); 
	
	const int N_x = p->N_x;
	const int N_y = p->N_y;
	const int N_z = p->N_z;
	const int log2N_x = p->log2N_x;
	const int log2N_y = p->log2N_y;
	
	const int sRxy = sizeof(REAL) * N_x * N_y;
	
	const int blocksx = N_x / 32 + 1;
	const int blocksy = N_y / 32 + 1;
	
	dim3 blocks(blocksx, blocksy);
	dim3 threads(32, 32);

	CUCOMPLEX* d_src  = p->d_A; //local vars for swapping workspace
	CUCOMPLEX* d_dest = p->d_B;
	CUCOMPLEX* d_tmp = 0;
	
	// first step is to copy in the spins, we will FT them aswell
	static const struct {
        double* h_s; //host memory
		CUCOMPLEX** d_s; //device memory
		CUCOMPLEX** d_h;
    } sd[] = { //sd = static data
        {sx, p->d_sx_q, p->d_hx_q},
        {sy, p->d_sy_q, p->d_hy_q},
        {sz, p->d_sz_q, p->d_hz_q}
	};

	for(int k=0; k<3; k++) // x y z
	{
		for(int z=0; z<N_z; z++)
		{
			d_src = p->d_A;
			d_dest = p->d_B;
			
			double* hs = &(sd[k].h_s[z*N_x*N_y]);
			CUCOMPLEX* ds = sd[k].d_s[z];
			
// 			start_timer(timer);
			memcpy(p->h_input, hs, sRxy);
			cudaMemcpy(p->d_input, p->h_input,sRxy, cudaMemcpyHostToDevice);
// 			stop_timer(timer);
			
			//twiddle to workspace
			twiddle2D<<<blocks, threads>>>(N_x, N_y, p->d_brp_x, p->d_brp_y, d_src, p->d_input);
			CHECK

			// FT spins
			d_src = p->d_A;
			d_dest = p->d_B;
			for(int i=0; i<log2N_x; i++)
			{
				Fi_2D_x<<<blocks, threads>>>(N_x, N_y, i, d_dest, d_src, p->d_exp2pi_x_f);
				CHECK

				d_tmp = d_src;
				d_src = d_dest;
				d_dest = d_tmp;
			}
			for(int i=0; i<log2N_y-1; i++)
			{
				Fi_2D_y<<<blocks, threads>>>(N_x, N_y, i, d_dest, d_src, p->d_exp2pi_y_f);
				CHECK

				d_tmp = d_src;
				d_src = d_dest;
				d_dest = d_tmp;
			}
			// last round for FT will land the spins in their proper destination (ds)
			Fi_2D_y<<<blocks, threads>>>(N_x, N_y, log2N_y-1, ds, d_src, p->d_exp2pi_y_f);
			CHECK
		}
	}
	
	// OK! Now we have all the spins FT'd and the interaction matrix ready.
	// WE will now convolve the signals into hq
	// zero the fields
	for(int k=0; k<3; k++) // x y z
	{
		CUCOMPLEX** hh = sd[k].d_h;
		for(int z=0; z<N_z; z++)
		{
			CUCOMPLEX* hhi = hh[z];
			setC<<<blocks, threads>>>(N_x, N_y, hhi, 0, 0);
			CHECK
		}
	}
	
	
	for(int targetLayer=0; targetLayer<N_z; targetLayer++)
	for(int sourceLayer=0; sourceLayer<N_z; sourceLayer++)
	{
		const int offset = (sourceLayer - targetLayer + N_z - 1);
		
		convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hx_q[targetLayer], p->d_sx_q[sourceLayer], p->d_GammaXX[offset]);
		convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hx_q[targetLayer], p->d_sy_q[sourceLayer], p->d_GammaXY[offset]);
		convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hx_q[targetLayer], p->d_sz_q[sourceLayer], p->d_GammaXZ[offset]);
// 
		convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hy_q[targetLayer], p->d_sx_q[sourceLayer], p->d_GammaXY[offset]);
		convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hy_q[targetLayer], p->d_sy_q[sourceLayer], p->d_GammaYY[offset]);
		convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hy_q[targetLayer], p->d_sz_q[sourceLayer], p->d_GammaYZ[offset]);
// 
		convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hz_q[targetLayer], p->d_sx_q[sourceLayer], p->d_GammaXZ[offset]);
		convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hz_q[targetLayer], p->d_sy_q[sourceLayer], p->d_GammaYZ[offset]);
		convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hz_q[targetLayer], p->d_sz_q[sourceLayer], p->d_GammaZZ[offset]);
		CHECK
	}

	// h(q) now calculated, iFT them and copy out to host
	static const struct {
        double* h_h; //host memory
		CUCOMPLEX** d_h; //device memory
    } sss[] = { //sd = static data
        {hx, p->d_hx_q},
        {hy, p->d_hy_q},
        {hz, p->d_hz_q}
	};
	
	for(int k=0; k<3; k++)
	{
		CUCOMPLEX** d_h = sss[k].d_h;
		for(int z=0; z<N_z; z++)
		{
			d_src = p->d_A;
			d_dest = p->d_B;
			// twiddle the fields
			twiddle2DCC<<<blocks, threads>>>(N_x, N_y, p->d_brp_x, p->d_brp_y, d_src, d_h[z]);
			CHECK
			
			//twiddled and waiting in d_src
			for(int i=0; i<log2N_x; i++)
			{
				Fi_2D_x<<<blocks, threads>>>(N_x, N_y, i, d_dest, d_src, p->d_exp2pi_x_b);

				d_tmp = d_src;
				d_src = d_dest;
				d_dest = d_tmp;
			}
			for(int i=0; i<log2N_y; i++)
			{
				Fi_2D_y<<<blocks, threads>>>(N_x, N_y, i, d_dest, d_src, p->d_exp2pi_y_b);

				d_tmp = d_src;
				d_src = d_dest;
				d_dest = d_tmp;
			}
			
			//real space of iFFT in d_src, need to chop off the (hopefully) zero imag part
			getRPart<<<blocks, threads>>>(N_x, N_y, p->d_output, d_src);
			CHECK
			
// 			start_timer(timer);
			//and copy it from the device to the host
			cudaMemcpy(p->h_output, p->d_output, sRxy, cudaMemcpyDeviceToHost);

			memcpy(&(sss[k].h_h[z*N_x*N_y]), p->h_output, sRxy);
// 			stop_timer(timer);
			
		}
	}
	
// 	printf("%f s\n", get_time(timer)); 
	//holy crap, we're done.
}

