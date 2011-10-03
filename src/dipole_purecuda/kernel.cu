#include <complex>
using namespace std;

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>
#include <stdio.h>

#define JM_FORWARD 1
#define JM_BACKWARD 0

#define REAL double
#define CUCOMPLEX cuDoubleComplex
#define MAKECOMPLEX(a,b) make_cuDoubleComplex(a,b) 

#define SMART_SCHEDULE

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
#define CHECK \
{ \
	const cudaError_t i = cudaGetLastError();\
	if(i) \
		printf("(%s:%i) %s\n",  __FILE__, __LINE__-1, cudaGetErrorString(i));\
}
#else
#define CHECK ;

#endif



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

__global__ void getRPart(const int N_x, const int N_y, REAL* d_dest,  CUCOMPLEX* d_src)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;

	const int idx = i + y * N_x;

#ifdef BOUND_CHECKS
	if(idx >= N_x * N_y) return;
#endif
	
	d_dest[idx] = d_src[idx].x;
}
__global__ void getIPart(const int N_x, const int N_y, REAL* d_dest,  CUCOMPLEX* d_src)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;

	const int idx = i + y * N_x;

#ifdef BOUND_CHECKS
	if(idx >= N_x * N_y) return;
#endif
	
	d_dest[idx] = d_src[idx].y;
}

__global__ void Fi_2D_x(const int N_x, const int N_y, const int step,  CUCOMPLEX* dst, const CUCOMPLEX* src, CUCOMPLEX* exp2pi)
{
	IDX_PATT(i, y);

	if(i >= N_x || y >= N_y)
		return;
	
	const unsigned int  bits = (2 << step) - 1;
	const unsigned int  half = (1 << step);
	const unsigned int hbits = half - 1;
	const unsigned int base  = (0xFFFF ^ bits);

	const int j   = i &  bits; // count in subset
	const int q   = i & hbits; // count in half set
	const int k   = i &  base; // base

	const int _a = k + y*N_x + j;
	const int _b = k + y*N_x + q;
	const int _c = step*N_x+i;
	const int _d = _b + half;
#ifdef BOUND_CHECKS
	if(_a >= N_x*N_y) return;
	if(_b >= N_x*N_y) return;
	if(_c >= N_x*N_y) return;
	if(_d >= N_x*N_y) return;
#endif
	
	dst[_a] = cuCadd(src[_b], cuCmul(exp2pi[_c], src[_d]));
}


__global__ void Fi_2D_y(const int N_x, const int N_y, const int step,  CUCOMPLEX* dst, const CUCOMPLEX* src, CUCOMPLEX* exp2pi)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;
	
	const unsigned int  bits = (2 << step) - 1;
	const unsigned int  half = (1 << step);
	const unsigned int hbits = half - 1;
	const unsigned int base  = (0xFFFF ^ bits);

	const int j   = y &  bits; // count in subset
	const int q   = y & hbits; // count in half set
	const int k   = y &  base; // base

	const int _a = (k+j)*N_x + i;
	const int _b = (k+q)*N_x + i;
	const int _c = step*N_y+y;
	const int _d = (k+q+half)*N_x + i;
	
#ifdef BOUND_CHECKS
	if(_a >= N_x*N_y) return;
	if(_b >= N_x*N_y) return;
	if(_c >= N_x*N_y) return;
	if(_d >= N_x*N_y) return;
#endif
	
	dst[_a] = cuCadd(src[_b], cuCmul(exp2pi[_c], src[_d]));
}

__global__ void twiddle2D(const int N_x, const int N_y, int* d_brp_x, int* d_brp_y, CUCOMPLEX* d_dest, const REAL* d_src)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;

	const int ti = d_brp_x[i];
	const int ty = d_brp_y[y];
	
	const int _a = ti + ty * N_x;
	const int _b = i + y * N_x;
	
#ifdef BOUND_CHECKS
	if(_a >= N_x*N_y) return;
	if(_b >= N_x*N_y) return;	
#endif
	
	d_dest[_a] = MAKECOMPLEX(d_src[_b], 0);
}

__global__ void twiddle1Dx_rc(const int N_x, const int N_y, int* d_brp_x, CUCOMPLEX* d_dest, const REAL* d_src)
{
	IDX_PATT(i, y);

	if(i >= N_x || y >= N_y)
		return;

	const int ti = d_brp_x[i];
	const int ty = y;
	
	const int _a = ti + ty * N_x;
	const int _b = i + y * N_x;
	
#ifdef BOUND_CHECKS
	if(_a >= N_x*N_y) return;
	if(_b >= N_x*N_y) return;	
#endif
	
	d_dest[_a].x = d_src[_b];
	d_dest[_a].y = 0;
}
__global__ void twiddle1Dx_cc(const int N_x, const int N_y, int* d_brp_x, CUCOMPLEX* d_dest, const CUCOMPLEX* d_src)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;

	const int ti = d_brp_x[i];
	const int ty = y;
	
	const int _a = ti + ty * N_x;
	const int _b = i + y * N_x;
	
#ifdef BOUND_CHECKS
	if(_a >= N_x*N_y) return;
	if(_b >= N_x*N_y) return;	
#endif
	
	d_dest[_a] = d_src[_b];
}

__global__ void twiddle1Dy_cc(const int N_x, const int N_y, int* d_brp_y, CUCOMPLEX* d_dest, const CUCOMPLEX* d_src)
{
	IDX_PATT(i, y);

	if(i >= N_x || y >= N_y)
		return;

	const int ti = i;
	const int ty = d_brp_y[y];
	
	const int _a = ti + ty * N_x;
	const int _b = i + y * N_x;
	
#ifdef BOUND_CHECKS
	if(_a >= N_x*N_y) return;
	if(_b >= N_x*N_y) return;	
#endif
	
	d_dest[_a] = d_src[_b];
}

__global__ void twiddle2DCC(const int N_x, const int N_y, int* d_brp_x, int* d_brp_y, CUCOMPLEX* d_dest,  const CUCOMPLEX* d_src)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;

	const int ti = d_brp_x[i];
	const int ty = d_brp_y[y];
	
	const int _a = ti + ty * N_x;
	const int _b = i + y * N_x;
	
#ifdef BOUND_CHECKS
	if(_a >= N_x*N_y) return;
	if(_b >= N_x*N_y) return;	
#endif
	
	d_dest[_a] = d_src[_b];
}

__global__ void scaleC(const int N_x, const int N_y, CUCOMPLEX* d, CUCOMPLEX* s,  double v)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;
	
	const int idx = i + y * N_x;
	
#ifdef BOUND_CHECKS
	if(idx >= N_x * N_y) return;
#endif
	
	d[idx].x = v * s[idx].x;
	d[idx].y = v * s[idx].y;
}
__global__ void setC(const int N_x, const int N_y, CUCOMPLEX* v,  double R, double I)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;
	
	const int idx = i + y * N_x;

#ifdef BOUND_CHECKS
	if(idx >= N_x * N_y) return;
#endif
	
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
	cudaHostAlloc(&h_brp_x, sizeof(int) * N_x, cudaHostAllocWriteCombined); CHECK//host "write-only"
	cudaHostAlloc(&h_brp_y, sizeof(int) * N_y, cudaHostAllocWriteCombined); CHECK//host "write-only"
	cudaMalloc(&(p->d_brp_x), sizeof(int) * N_x);CHECK
	cudaMalloc(&(p->d_brp_y), sizeof(int) * N_y);CHECK

	for(int i=0; i<N_x; i++)
		h_brp_x[i] = bit_rev_perm(i, log2N_x);
	for(int i=0; i<N_y; i++)
		h_brp_y[i] = bit_rev_perm(i, log2N_y);

	cudaMemcpy(p->d_brp_x, h_brp_x, sizeof(int) * N_x, cudaMemcpyHostToDevice);	CHECK
	cudaMemcpy(p->d_brp_y, h_brp_y, sizeof(int) * N_y, cudaMemcpyHostToDevice);	CHECK
	cudaFreeHost(h_brp_x);	CHECK
	cudaFreeHost(h_brp_y);	CHECK

	CUCOMPLEX* h_exp2pi_x;
	CUCOMPLEX* h_exp2pi_y; 
	cudaHostAlloc(&h_exp2pi_x, sizeof(CUCOMPLEX) * N_x * log2N_x, cudaHostAllocWriteCombined);	CHECK //host "write-only"
	cudaHostAlloc(&h_exp2pi_y, sizeof(CUCOMPLEX) * N_y * log2N_y, cudaHostAllocWriteCombined);	CHECK //host "write-only"
	cudaMalloc(&(p->d_exp2pi_x_f), sizeof(CUCOMPLEX) * N_x * log2N_x);	CHECK
	cudaMalloc(&(p->d_exp2pi_y_f), sizeof(CUCOMPLEX) * N_y * log2N_y);	CHECK
	cudaMalloc(&(p->d_exp2pi_x_b), sizeof(CUCOMPLEX) * N_x * log2N_x);	CHECK
	cudaMalloc(&(p->d_exp2pi_y_b), sizeof(CUCOMPLEX) * N_y * log2N_y);	CHECK

	for(int j=0; j<log2N_x; j++)
	{
		for(int i=0; i<N_x; i++)
		{
			complex<double> x = get_exp(j, i, JM_FORWARD);
// 			if(j==2) printf("### %i   %g %g\n", i, x.real(), x.imag());
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
	
	cudaMemcpy(p->d_exp2pi_x_f, h_exp2pi_x, sizeof(CUCOMPLEX) * N_x * log2N_x, cudaMemcpyHostToDevice);	CHECK
	cudaMemcpy(p->d_exp2pi_y_f, h_exp2pi_y, sizeof(CUCOMPLEX) * N_y * log2N_y, cudaMemcpyHostToDevice);	CHECK

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
	
	cudaMemcpy(p->d_exp2pi_x_b, h_exp2pi_x, sizeof(CUCOMPLEX) * N_x * log2N_x, cudaMemcpyHostToDevice);	CHECK
	cudaMemcpy(p->d_exp2pi_y_b, h_exp2pi_y, sizeof(CUCOMPLEX) * N_y * log2N_y, cudaMemcpyHostToDevice);	CHECK
	
	cudaFreeHost(h_exp2pi_x);	CHECK
	cudaFreeHost(h_exp2pi_y);	CHECK
	// DONE making parts needed for future FFTs
	
	cudaMalloc(&(p->d_A), sCxy);	CHECK
	cudaMalloc(&(p->d_B), sCxy);	CHECK
	

	// 2D arrays, 1st dimmension is for layer
	p->d_sx_q = new CUCOMPLEX*[nz];
	p->d_sy_q = new CUCOMPLEX*[nz];
	p->d_sz_q = new CUCOMPLEX*[nz];

	p->d_hx_q = new CUCOMPLEX*[nz];
	p->d_hy_q = new CUCOMPLEX*[nz];
	p->d_hz_q = new CUCOMPLEX*[nz];

	for(int i=0; i<nz; i++)
	{
		cudaMalloc(&(p->d_sx_q[i]), sCxy);		CHECK
		cudaMalloc(&(p->d_sy_q[i]), sCxy);		CHECK
		cudaMalloc(&(p->d_sz_q[i]), sCxy);		CHECK
	
		cudaMalloc(&(p->d_hx_q[i]), sCxy);		CHECK
		cudaMalloc(&(p->d_hy_q[i]), sCxy);		CHECK
		cudaMalloc(&(p->d_hz_q[i]), sCxy);		CHECK
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
		cudaMalloc(&(p->d_GammaXX[i]), sCxy);		CHECK
		cudaMalloc(&(p->d_GammaXY[i]), sCxy);		CHECK
		cudaMalloc(&(p->d_GammaXZ[i]), sCxy);		CHECK
	
		cudaMalloc(&(p->d_GammaYY[i]), sCxy);		CHECK
		cudaMalloc(&(p->d_GammaYZ[i]), sCxy);		CHECK
	
		cudaMalloc(&(p->d_GammaZZ[i]), sCxy);		CHECK
	}
	
	
	cudaHostAlloc(&(p->h_input), sRxy, cudaHostAllocWriteCombined); CHECK // "write-only"
	cudaHostAlloc(&(p->h_output),sRxy, 0);	CHECK

	cudaMalloc(&(p->d_input), sRxy);	CHECK
	cudaMalloc(&(p->d_output),sRxy);	CHECK
	
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

	
	#ifdef SMART_SCHEDULE
	//different thread schedules for different access patterns
	dim3 blocksx(N_y);
	dim3 blocksy(N_x);
	dim3 threadsxx(N_x);
	dim3 threadsyy(N_y);
	#else
	const int _blocksx = N_x / 32 + 1;
	const int _blocksy = N_y / 32 + 1;
	dim3 blocksx(_blocksx, _blocksy);
	dim3 blocksy(_blocksx, _blocksy);
	dim3 threadsxx(32,32);
	dim3 threadsyy(32,32);
	#endif
	
	CUCOMPLEX* d_src  = p->d_A; //local vars for swapping workspace
	CUCOMPLEX* d_dest = p->d_B;
	CUCOMPLEX* d_tmp = 0;
	
	for(int k=0; k<6; k++) //XX XY XZ   YY YZ   ZZ
	{
		for(int j=0; j<nz; j++)
		{
			d_src = p->d_A;
			d_dest = p->d_B;
			
			memcpy(p->h_input, &(sd[k].h[j*N_x*N_y]), sRxy);
			cudaMemcpy(p->d_input, p->h_input,sRxy, cudaMemcpyHostToDevice);
			CHECK

			// twiddle the data, next step will be to fft it. 
			// twiddle2D<<<blocks, threads>>>(N_x, N_y, p->d_brp_x, p->d_brp_y, p->d_A, p->d_input);
			// CHECK

			twiddle1Dx_rc<<<blocksx, threadsxx>>>(N_x, N_y, p->d_brp_x, d_dest, p->d_input);
			CHECK
			twiddle1Dy_cc<<<blocksy, threadsyy>>>(N_x, N_y, p->d_brp_y, d_src, d_dest);
			CHECK

			for(int i=0; i<log2N_x; i++)
			{
				Fi_2D_x<<<blocksx, threadsxx>>>(N_x, N_y, i, d_dest, d_src, p->d_exp2pi_x_f);
				CHECK
			
				d_tmp = d_src;
				d_src = d_dest;
				d_dest = d_tmp;
			}
			for(int i=0; i<log2N_y; i++)
			{
				Fi_2D_y<<<blocksy, threadsyy>>>(N_x, N_y, i, d_dest, d_src, p->d_exp2pi_y_f);
				CHECK

				d_tmp = d_src;
				d_src = d_dest;
				d_dest = d_tmp;
			}
			
			// going to prescale the data into d_GammaAB:
			scaleC<<<blocksx, threadsxx>>>(N_x, N_y, sd[k].d[j], d_src, 1.0/((double)(N_x * N_y)));
			CHECK
		}
	}
	
	return p;
}

void free_JM_LONGRANGE_PLAN(JM_LONGRANGE_PLAN* p)
{
	const int N_z = p->N_z;
	const int nz = N_z * 2 - 1;
	
	cudaFree(p->d_brp_x); CHECK
	cudaFree(p->d_brp_y); CHECK
	
	cudaFree(p->d_exp2pi_x_f); CHECK
	cudaFree(p->d_exp2pi_y_f); CHECK
	cudaFree(p->d_exp2pi_x_b); CHECK
	cudaFree(p->d_exp2pi_y_b); CHECK

	cudaFree(p->d_input); CHECK
	cudaFree(p->d_output); CHECK
	
	cudaFreeHost(p->h_input); CHECK
	cudaFreeHost(p->h_output); CHECK
	
	for(int z=0; z<N_z; z++)
	{
		cudaFree(p->d_sx_q[z]); CHECK
		cudaFree(p->d_sy_q[z]); CHECK
		cudaFree(p->d_sz_q[z]); CHECK
	
		cudaFree(p->d_hx_q[z]); CHECK
		cudaFree(p->d_hy_q[z]); CHECK
		cudaFree(p->d_hz_q[z]); CHECK
	}
	
	delete [] p->d_sx_q;
	delete [] p->d_sy_q;
	delete [] p->d_sz_q;

	delete [] p->d_hx_q;
	delete [] p->d_hy_q;
	delete [] p->d_hz_q;

	for(int z=0; z<nz; z++)
	{
		cudaFree(p->d_GammaXX[z]); CHECK
		cudaFree(p->d_GammaXY[z]); CHECK
		cudaFree(p->d_GammaXZ[z]); CHECK
		
		cudaFree(p->d_GammaYY[z]); CHECK
		cudaFree(p->d_GammaYZ[z]); CHECK
		
		cudaFree(p->d_GammaZZ[z]); CHECK
	}
	
	
	delete [] p->d_GammaXX;
	delete [] p->d_GammaXY;
	delete [] p->d_GammaXZ;
	
	delete [] p->d_GammaYY;
	delete [] p->d_GammaYZ;
	
	delete [] p->d_GammaZZ;
	
	cudaFree(p->d_A); CHECK
	cudaFree(p->d_B); CHECK

	delete p;
}

__global__ void convolveSum(const int N_x, const int N_y, CUCOMPLEX* d_dest,  CUCOMPLEX* d_A,  CUCOMPLEX* d_B)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;
	
	const int idx = i + y * N_x;

#ifdef BOUND_CHECKS
	if(idx >= N_x * N_y) return;
#endif
	
	d_dest[idx] = cuCadd(d_dest[idx], cuCmul(d_A[idx], d_B[idx]));
}


__global__ void getLayer(const int nx, const int ny, const int layer, REAL* d_dest,  const REAL* d_src)
{
	IDX_PATT(row, col);
	
	if(row >= nx || col >= ny)
		return;

	const int _a = col + row*nx;
	const int _b = _a + layer*nx*ny;
	
	d_dest[_a] = d_src[_b];
}

__global__ void setLayer(const int N_x, const int N_y, const int layer, REAL* d_dest,  REAL* d_src)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;


	const int _a = i + y * N_x + layer * N_x * N_y;
	const int _b = i + y * N_x;
#ifdef BOUND_CHECKS
	if(_a >= N_x * N_y) return;
	if(_b >= N_x * N_y) return;
#endif
	
	d_dest[_a] = d_src[_b];
}

void JM_LONGRANGE(JM_LONGRANGE_PLAN* p, 
				  const double* d_sx, const double* d_sy, const double* d_sz,
				  double* d_hx, double* d_hy, double* d_hz)
{
	const int N_x = p->N_x;
	const int N_y = p->N_y;
	const int N_z = p->N_z;
	const int log2N_x = p->log2N_x;
	const int log2N_y = p->log2N_y;
	const int sRxy = sizeof(REAL)*N_x*N_y;
	
	

	#ifdef SMART_SCHEDULE
	//different thread schedules for different access patterns
	dim3 blocksx(N_y);
	dim3 blocksy(N_x);
	dim3 threadsxx(N_x);
	dim3 threadsyy(N_y);
	#else
	const int _blocksx = N_x / 32 + 1;
	const int _blocksy = N_y / 32 + 1;
	dim3 blocksx(_blocksx, _blocksy);
	dim3 blocksy(_blocksx,_blocksy);
	dim3 threadsxx(32,32);
	dim3 threadsyy(32,32);
	#endif

	CUCOMPLEX* d_src  = p->d_A; //local vars for swapping workspace
	CUCOMPLEX* d_dest = p->d_B;
	CUCOMPLEX* d_tmp = 0;
	
	// FT the spins
	static const struct {
        const double*	d_s_r;
		CUCOMPLEX** 	d_s_q;
		CUCOMPLEX** 	d_h_q;
		double*     	d_h_r;
    } sd[] = { //sd = static data
        {d_sx, p->d_sx_q, p->d_hx_q, d_hx},
        {d_sy, p->d_sy_q, p->d_hy_q, d_hy},
        {d_sz, p->d_sz_q, p->d_hz_q, d_hz}
	};

	for(int k=0; k<3; k++) // x y z
	{
		const double* d_s_r = sd[k].d_s_r;
		
		for(int z=0; z<N_z; z++)
		{
			d_src = p->d_A;
			d_dest = p->d_B;
			
			CUCOMPLEX* d_s_q = sd[k].d_s_q[z];
			
			//grab a layer
			getLayer<<<blocksx, threadsxx>>>(N_x, N_y, z, p->d_input, d_s_r);
			CHECK

			//twiddle to workspace
			//twiddle2D<<<blocks, threads>>>(N_x, N_y, p->d_brp_x, p->d_brp_y, d_src, p->d_input);
			//CHECK

			twiddle1Dx_rc<<<blocksx, threadsxx>>>(N_x, N_y, p->d_brp_x, d_dest, p->d_input);
			CHECK
			twiddle1Dy_cc<<<blocksy, threadsyy>>>(N_x, N_y, p->d_brp_y, d_src, d_dest);
			CHECK

			// FT spins
			for(int step=0; step<log2N_x; step++)
			{
				Fi_2D_x<<<blocksx, threadsxx>>>(N_x, N_y, step, d_dest, d_src, p->d_exp2pi_x_f);
				CHECK

				d_tmp = d_src;
				d_src = d_dest;
				d_dest = d_tmp;
			}
			for(int step=0; step<log2N_y-1; step++)
			{
 				Fi_2D_y<<<blocksy, threadsyy>>>(N_x, N_y, step, d_dest, d_src, p->d_exp2pi_y_f);
 				CHECK

				d_tmp = d_src;
				d_src = d_dest;
				d_dest = d_tmp;
			}
			// last round for FT will land the spins in their proper destination (d_s_q)
			Fi_2D_y<<<blocksy, threadsyy>>>(N_x, N_y, log2N_y-1, d_s_q, d_src, p->d_exp2pi_y_f);
			CHECK 
		}
	}

	// OK! Now we have all the spins FT'd and the interaction matrix ready.
	// We will now convolve the signals into hq
	// zero the fields
	for(int k=0; k<3; k++) // x y z
	{
		CUCOMPLEX** hh = sd[k].d_h_q;
		for(int z=0; z<N_z; z++)
		{
			CUCOMPLEX* hhi = hh[z];
			setC<<<blocksx, threadsxx>>>(N_x, N_y, hhi, 0, 0);
			CHECK
		}
	}
	
	

	for(int targetLayer=0; targetLayer<N_z; targetLayer++)
	for(int sourceLayer=0; sourceLayer<N_z; sourceLayer++)
	{
		const int offset = (sourceLayer - targetLayer + N_z - 1);
		convolveSum<<<blocksx, threadsxx>>>(N_x, N_y, p->d_hx_q[targetLayer], p->d_sx_q[sourceLayer], p->d_GammaXX[offset]);
		convolveSum<<<blocksx, threadsxx>>>(N_x, N_y, p->d_hx_q[targetLayer], p->d_sy_q[sourceLayer], p->d_GammaXY[offset]);
		convolveSum<<<blocksx, threadsxx>>>(N_x, N_y, p->d_hx_q[targetLayer], p->d_sz_q[sourceLayer], p->d_GammaXZ[offset]);

		convolveSum<<<blocksx, threadsxx>>>(N_x, N_y, p->d_hy_q[targetLayer], p->d_sx_q[sourceLayer], p->d_GammaXY[offset]);
		convolveSum<<<blocksx, threadsxx>>>(N_x, N_y, p->d_hy_q[targetLayer], p->d_sy_q[sourceLayer], p->d_GammaYY[offset]);
		convolveSum<<<blocksx, threadsxx>>>(N_x, N_y, p->d_hy_q[targetLayer], p->d_sz_q[sourceLayer], p->d_GammaYZ[offset]);

		convolveSum<<<blocksx, threadsxx>>>(N_x, N_y, p->d_hz_q[targetLayer], p->d_sx_q[sourceLayer], p->d_GammaXZ[offset]);
		convolveSum<<<blocksx, threadsxx>>>(N_x, N_y, p->d_hz_q[targetLayer], p->d_sy_q[sourceLayer], p->d_GammaYZ[offset]);
		convolveSum<<<blocksx, threadsxx>>>(N_x, N_y, p->d_hz_q[targetLayer], p->d_sz_q[sourceLayer], p->d_GammaZZ[offset]);
		CHECK
	}

	// h(q) now calculated, iFT them 
	for(int k=0; k<3; k++) //for each component
	{
		CUCOMPLEX** d_h = sd[k].d_h_q;
		double* d_hxyz = sd[k].d_h_r;
		for(int z=0; z<N_z; z++) //for each layer
		{
			d_src = p->d_A;
			d_dest = p->d_B;
			// twiddle the fields
			//twiddle2DCC<<<blocks, threads>>>(N_x, N_y, p->d_brp_x, p->d_brp_y, d_src, d_h[z]);
			//CHECK

			twiddle1Dx_cc<<<blocksx, threadsxx>>>(N_x, N_y, p->d_brp_x, d_dest, d_h[z]);
			CHECK
			twiddle1Dy_cc<<<blocksy, threadsyy>>>(N_x, N_y, p->d_brp_y, d_src, d_dest);
			CHECK

			//twiddled and waiting in d_src
			for(int step=0; step<log2N_x; step++)
			{
				Fi_2D_x<<<blocksx, threadsxx>>>(N_x, N_y, step, d_dest, d_src, p->d_exp2pi_x_b);
				CHECK

				d_tmp = d_src;
				d_src = d_dest;
				d_dest = d_tmp;
			}
			for(int step=0; step<log2N_y; step++)
			{
				Fi_2D_y<<<blocksy, threadsyy>>>(N_x, N_y, step, d_dest, d_src, p->d_exp2pi_y_b);
				CHECK

				d_tmp = d_src;
				d_src = d_dest;
				d_dest = d_tmp;
			}
			
			//real space of iFFT in d_src, need to chop off the (hopefully) zero imag part
			getRPart<<<blocksx, threadsxx>>>(N_x, N_y, p->d_output, d_src);
			CHECK
			
			setLayer<<<blocksx, threadsxx>>>(N_x, N_y, z, d_hxyz, p->d_output);
			CHECK
		}
	}
	//holy crap, we're done.
}

