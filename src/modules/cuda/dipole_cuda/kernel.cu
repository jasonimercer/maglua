#include <complex>
#include <iostream>
#include <vector>
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

// #define SMART_SCHEDULE

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

__global__ void getRPart(const int N_x, const int N_y, REAL* d_dest,  CUCOMPLEX* d_src)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;

	const int idx = i + y * N_x;

	d_dest[idx] = d_src[idx].x;
}
__global__ void getIPart(const int N_x, const int N_y, REAL* d_dest,  CUCOMPLEX* d_src)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;

	const int idx = i + y * N_x;

	d_dest[idx] = d_src[idx].y;
}


__global__ void scaleC(const int N_x, const int N_y, CUCOMPLEX* d, CUCOMPLEX* s,  double v)
{
	IDX_PATT(x, y);
	
	if(x >= N_x || y >= N_y)
		return;
	
	const int idx = x + y * N_x;
	
// 	d[idx] = cuCmul(MAKECOMPLEX(v,0), s[idx]);
	d[idx].x = v * s[idx].x;
	d[idx].y = v * s[idx].y;
}
static void d_scaleC(const int nx, const int ny, CUCOMPLEX* d_dest, CUCOMPLEX* d_src, double scale)
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
	
	scaleC<<<blocks, threads>>>(nx, ny, d_dest, d_src, scale);
	KCHECK;
}




__global__ void setC(const int N_x, const int N_y, CUCOMPLEX* v,  double R, double I)
{
	IDX_PATT(x, y);
	
	if(x >= N_x || y >= N_y)
		return;
	
	const int idx = x + y * N_x;

	v[idx].x = R;
	v[idx].y = I;
}




typedef struct JM_LONGRANGE_PLAN
{
	int N_x, N_y, N_z;
	vector<int> Rx; //factors for N_x
	vector<int> Ry; //factors for N_y
	
	CUCOMPLEX* d_exp2pi_x_f; //this will hold exp terms
	CUCOMPLEX* d_exp2pi_y_f;
	CUCOMPLEX* d_exp2pi_x_b; //backward
	CUCOMPLEX* d_exp2pi_y_b;
	
	// base and steps used in ffts
	int* d_base_x;
	int* d_base_y;
	int* d_step_x;
	int* d_step_y;	
	
	REAL* d_output;
	
	CUCOMPLEX* h_temp; 
	
	// 2D arrays, 1st dimmension is for layer
	CUCOMPLEX** d_sx_q;
	CUCOMPLEX** d_sy_q;
	CUCOMPLEX** d_sz_q;
	
	CUCOMPLEX*  d_hA_q;
	
	// 2D arrays, 1st dimmension is for interlayer offset
	CUCOMPLEX** d_GammaXX;
	CUCOMPLEX** d_GammaXY;
	CUCOMPLEX** d_GammaXZ;
	
	CUCOMPLEX** d_GammaYY;
	CUCOMPLEX** d_GammaYZ;
	
	CUCOMPLEX** d_GammaZZ;
	
	CUCOMPLEX* d_A; //chunks of memory for out-of-place FFT calcs, workscapes
	CUCOMPLEX* d_B;
}JM_LONGRANGE_PLAN;


// mixed radix plan
typedef struct s_if
{
	int i; //base
	int s; //step
	
	complex<double> e; //exp term
} s_if;

typedef struct fft_plan
{
	s_if** p;
	int n;
	vector<int> R;
} fft_plan;


// this is where all the "magic" happens. This records the important info
// as it virutally calculates an fft recursively. 
static void fft_R(vector<int> idx, vector<int> R, int step, double sign, s_if** plan)
{
	int r = R[step];
	
	int denom = 1;
	for(unsigned int i=step; i<R.size(); i++)
		denom *= R[i];
	
	vector<int>* t = new vector<int>[r];
	
	for(unsigned int i=0; i<idx.size(); i++)
	{
		t[i%r].push_back(idx[i]);
	}
	
	if(idx.size() > r)
	for(int i=0; i<r; i++)
	{
		fft_R(t[i], R, step+1, sign, plan);
	}
	
	for(unsigned int i=0; i<idx.size(); i++)
	{
		int j = i % (idx.size() / r);

		plan[step][idx[i]].i = t[0][j];
		plan[step][idx[i]].s = t[1][j] - t[0][j];
		double fraction = ((double)i) / ((double)denom);
		//printf("step = %i, i=%i  frac=%i/%i\n", step, i, i, denom);
		plan[step][idx[i]].e = exp(sign * -2.0 * 3.14159265358979323846264 * complex<double>(0,1) * fraction);
	}
	
	delete [] t;
}

// attempt to factor a number into small primes
static int factor(int v, vector<int>& f)
{
	const int primes[7] = {2,3,5,7,11,13,17};
	
	f.clear();
	
	for(int p=0; p<7; p++)
	{
		const int d = primes[p];
		while(v%d == 0)
		{
			f.push_back(d);
			v /= d;
		}
	}
	
	return v == 1;
}

static fft_plan* make_plan(int n, int dir)
{
	vector<int> r;
	if(!factor(n, r))
	{
		return 0; //can't factor
	}
	
	s_if** plan = new s_if* [r.size()];
	for(unsigned int i=0; i<r.size(); i++)
	{
		plan[i] = new s_if[n];
		for(unsigned int j=0; j<n; j++)
		{
			plan[i][j].i = -1;
		}
	}
	
	vector<int> idx;
	for(int i=0; i<n; i++)
	{
		idx.push_back(i);
	}
	
	
	double sign = 1.0;
	if(dir < 0)
		sign = -1.0;
		
	fft_R(idx, r, 0, sign, plan);
	
	fft_plan* fplan = new fft_plan;
	fplan->p = plan;
	fplan->R = r;
	fplan->n = n;
	
	return fplan;
}

static void free_plan(fft_plan* p)
{
	if(!p) return;
	for(unsigned int i=0; i<p->R.size(); i++)
	{
		delete [] p->p[i];
	}
	delete [] p->p;
	delete p;
}

__global__ void __r2c(const int nx, const int ny, CUCOMPLEX* d_dest, const REAL* d_src)
{
	IDX_PATT(x, y);

	if(x >= nx || y >= ny)
		return;
	
	d_dest[x+y*nx] = MAKECOMPLEX(d_src[x+y*nx], 0);
}
static void d_r2c(const int nx, const int ny, CUCOMPLEX* d_dest, const REAL* d_src)
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
	
	__r2c<<<blocks, threads>>>(nx, ny, d_dest, d_src);
	KCHECK;
}


template <unsigned int radix>
__global__ void Fi_2D_x(const int nx, const int ny, 
			const CUCOMPLEX* exp2pi, const int* bases, const int* steps, const int offset,
			CUCOMPLEX* dest, const CUCOMPLEX* src)
{
	IDX_PATT(x, y);

	if(x >= nx || y >= ny)
		return;
	
	const int idx  = x + y*nx;
	const int base = bases[offset + x];
	const int step = steps[offset + x];
			
	dest[idx] = MAKECOMPLEX(0,0);
#pragma unroll
	for(int j=radix-1; j>=0; j--)
	{
		dest[idx] = cuCmul(dest[idx], exp2pi[offset + x]);
		dest[idx] = cuCadd(dest[idx], src[(base + step * j) + y*nx]);
	}
}

template <unsigned int radix>
__global__ void Fi_2D_y(const int nx, const int ny, 
			const CUCOMPLEX* exp2pi, const int* bases, const int* steps, const int offset,
			CUCOMPLEX* dest, const CUCOMPLEX* src)
{
	IDX_PATT(x, y);

	if(x >= nx || y >= ny)
		return;
	
	const int idx  = x + y*nx;
	const int base = bases[offset + y];
	const int step = steps[offset + y];
	
	dest[idx] = MAKECOMPLEX(0,0);
#pragma unroll
	for(int j=radix-1; j>=0; j--)
	{
		dest[idx] = cuCmul(dest[idx], exp2pi[offset + y]);
		dest[idx] = cuCadd(dest[idx], src[x + (base + step * j)*nx]);
	}
}

static void fourier2D(int nx, int ny, vector<int>& rx, vector<int>& ry, 
		      CUCOMPLEX* d_exp2pi_x, CUCOMPLEX* d_exp2pi_y, 
		      int* d_base_x, int* d_base_y, 
		      int* d_step_x, int* d_step_y,
		      CUCOMPLEX* d_dest, CUCOMPLEX* d_src)
{
	#ifdef SMART_SCHEDULE
	dim3 blocks(nx);
	dim3 threads(ny);
	#else
	const int _blocksx = nx / 32 + 1;
	const int _blocksy = ny / 32 + 1;
	dim3 blocks(_blocksx, _blocksy);
	dim3 threads(32,32);
	#endif	
	
	CUCOMPLEX* A = d_src;
	CUCOMPLEX* B = d_dest;
	CUCOMPLEX* T;

	// X direction
	for(int i=rx.size()-1; i>=0; i--)
	{
		switch(rx[i])
		{
		case  2: Fi_2D_x< 2><<<blocks, threads>>>(nx, ny, d_exp2pi_x, d_base_x, d_step_x, i*nx, B, A); break;
		case  3: Fi_2D_x< 3><<<blocks, threads>>>(nx, ny, d_exp2pi_x, d_base_x, d_step_x, i*nx, B, A); break;
		case  5: Fi_2D_x< 5><<<blocks, threads>>>(nx, ny, d_exp2pi_x, d_base_x, d_step_x, i*nx, B, A); break;
		case  7: Fi_2D_x< 7><<<blocks, threads>>>(nx, ny, d_exp2pi_x, d_base_x, d_step_x, i*nx, B, A); break;
		case 11: Fi_2D_x<11><<<blocks, threads>>>(nx, ny, d_exp2pi_x, d_base_x, d_step_x, i*nx, B, A); break;
		case 13: Fi_2D_x<13><<<blocks, threads>>>(nx, ny, d_exp2pi_x, d_base_x, d_step_x, i*nx, B, A); break;
		case 17: Fi_2D_x<17><<<blocks, threads>>>(nx, ny, d_exp2pi_x, d_base_x, d_step_x, i*nx, B, A); break;
		default:
		    fprintf(stderr, "(%s:%i) Spurious radix: %i\n", __FILE__, __LINE__, rx[i]);
		}
		KCHECK;
		T = A;
		A = B;
		B = T;
	}

	// Y direction
	for(int i=ry.size()-1; i>=0; i--)
	{
		switch(ry[i])
		{
		case  2: Fi_2D_y< 2><<<blocks, threads>>>(nx, ny, d_exp2pi_y, d_base_y, d_step_y, i*ny, B, A); break;
		case  3: Fi_2D_y< 3><<<blocks, threads>>>(nx, ny, d_exp2pi_y, d_base_y, d_step_y, i*ny, B, A); break;
		case  5: Fi_2D_y< 5><<<blocks, threads>>>(nx, ny, d_exp2pi_y, d_base_y, d_step_y, i*ny, B, A); break;
		case  7: Fi_2D_y< 7><<<blocks, threads>>>(nx, ny, d_exp2pi_y, d_base_y, d_step_y, i*ny, B, A); break;
		case 11: Fi_2D_y<11><<<blocks, threads>>>(nx, ny, d_exp2pi_y, d_base_y, d_step_y, i*ny, B, A); break;
		case 13: Fi_2D_y<13><<<blocks, threads>>>(nx, ny, d_exp2pi_y, d_base_y, d_step_y, i*ny, B, A); break;
		case 17: Fi_2D_y<17><<<blocks, threads>>>(nx, ny, d_exp2pi_y, d_base_y, d_step_y, i*ny, B, A); break;
		default:
		    fprintf(stderr, "(%s:%i) Spurious radix: %i\n", __FILE__, __LINE__, ry[i]);
		}
		KCHECK;
		T = A;
		A = B;
		B = T;
	}

	if(A != d_dest)
	    CHECKCALL(cudaMemcpy(d_dest, A, sizeof(CUCOMPLEX)*nx*ny, cudaMemcpyDeviceToDevice));
}

JM_LONGRANGE_PLAN* make_JM_LONGRANGE_PLAN(int N_x, int N_y, int N_z, 
	double* GammaXX, double* GammaXY, double* GammaXZ,
	                 double* GammaYY, double* GammaYZ,
	                                  double* GammaZZ)
{
	const int nz   = N_z;
	const int nxy = N_x * N_y;
	const int sRxy = sizeof(REAL) * nxy;
	const int sCxy = sizeof(CUCOMPLEX) * nxy;

	JM_LONGRANGE_PLAN* p = new JM_LONGRANGE_PLAN;

	if(!factor(N_x, p->Rx))
	{
		delete p;
		return 0;
	}
	if(!factor(N_y, p->Ry))
	{
		delete p;
		return 0;
	}
	
	p->N_x = N_x;
	p->N_y = N_y;
	p->N_z = N_z;
	
	fft_plan* forward_x = make_plan(N_x, 1);
	fft_plan* forward_y = make_plan(N_y, 1);

	fft_plan* backward_x = make_plan(N_x, -1);
	fft_plan* backward_y = make_plan(N_y, -1);
	
/*
	fft_plan* q = forward_x;

	for(int w=0; w<2; w++)
	{
	    cout << "w = " << w << endl;
	    for(int i=0; i<N_x; i++)
	    {
		cout << ")))) " << q->p[w][i].i << endl;
		cout << ")))) " << q->p[w][i].s << endl;
		cout << ">>>> " << q->p[w][i].e << endl;
	    }
	}
*/

// 	for(int i=0; i<4; i++)
// 	{
// 	    printf("#### %f\n", GammaXX[i]);
// 	}

	const int sRx = p->Rx.size();
	const int sRy = p->Ry.size();
	CUCOMPLEX* h_exp2pi_x_f;
	CUCOMPLEX* h_exp2pi_x_b;
	CUCOMPLEX* h_exp2pi_y_f;
	CUCOMPLEX* h_exp2pi_y_b;
	int* h_base_x;
	int* h_base_y;
	int* h_step_x;
	int* h_step_y;

	CHECKCALL(cudaMallocHost(&h_exp2pi_x_f, sizeof(CUCOMPLEX) * N_x * sRx));
	CHECKCALL(cudaMallocHost(&h_exp2pi_x_b, sizeof(CUCOMPLEX) * N_x * sRx));
	CHECKCALL(cudaMallocHost(&h_exp2pi_y_f, sizeof(CUCOMPLEX) * N_y * sRy));
	CHECKCALL(cudaMallocHost(&h_exp2pi_y_b, sizeof(CUCOMPLEX) * N_y * sRy));

	CHECKCALL(cudaMallocHost(&h_base_x, sizeof(int) * N_x * sRx));
	CHECKCALL(cudaMallocHost(&h_step_x, sizeof(int) * N_x * sRx));
	CHECKCALL(cudaMallocHost(&h_base_y, sizeof(int) * N_y * sRy));
	CHECKCALL(cudaMallocHost(&h_step_y, sizeof(int) * N_y * sRy));
	
	CHECKCALL(cudaMalloc(&(p->d_exp2pi_x_f), sizeof(CUCOMPLEX) * N_x * sRx));
	CHECKCALL(cudaMalloc(&(p->d_exp2pi_x_b), sizeof(CUCOMPLEX) * N_x * sRx));
	CHECKCALL(cudaMalloc(&(p->d_exp2pi_y_f), sizeof(CUCOMPLEX) * N_y * sRy));
	CHECKCALL(cudaMalloc(&(p->d_exp2pi_y_b), sizeof(CUCOMPLEX) * N_y * sRy));
	
	CHECKCALL(cudaMalloc(&(p->d_base_x), sizeof(int) * N_x * sRx));
	CHECKCALL(cudaMalloc(&(p->d_step_x), sizeof(int) * N_x * sRx));
	CHECKCALL(cudaMalloc(&(p->d_base_y), sizeof(int) * N_y * sRy));
	CHECKCALL(cudaMalloc(&(p->d_step_y), sizeof(int) * N_y * sRy));

	int c = 0;
	for(int j=0; j<sRx; j++)
	{
		for(int i=0; i<N_x; i++)
		{
		    complex<REAL> d;
		    d = forward_x->p[j][i].e;
		    h_exp2pi_x_f[c] = MAKECOMPLEX(d.real(), d.imag());
		    
		    d = backward_x->p[j][i].e;
		    h_exp2pi_x_b[c] = MAKECOMPLEX(d.real(), d.imag());
		    h_base_x[c] = forward_x->p[j][i].i;
		    h_step_x[c] = forward_x->p[j][i].s;
		    c++;
		}
	}
	CHECKCALL(cudaMemcpy(p->d_exp2pi_x_f, h_exp2pi_x_f, sizeof(CUCOMPLEX) * N_x * sRx, cudaMemcpyHostToDevice));
	CHECKCALL(cudaMemcpy(p->d_exp2pi_x_b, h_exp2pi_x_b, sizeof(CUCOMPLEX) * N_x * sRx, cudaMemcpyHostToDevice));
	CHECKCALL(cudaMemcpy(p->d_base_x, h_base_x, sizeof(int) * N_x * sRx, cudaMemcpyHostToDevice));
	CHECKCALL(cudaMemcpy(p->d_step_x, h_step_x, sizeof(int) * N_x * sRx, cudaMemcpyHostToDevice));

	c = 0;
	for(int j=0; j<sRy; j++)
	{
		for(int i=0; i<N_y; i++)
		{
		    complex<REAL> d;
		    d = forward_y->p[j][i].e;
		    h_exp2pi_y_f[c] = MAKECOMPLEX(d.real(), d.imag());
		    
		    d = backward_y->p[j][i].e;
		    h_exp2pi_y_b[c] = MAKECOMPLEX(d.real(), d.imag());
		    h_base_y[c] = forward_y->p[j][i].i;
		    h_step_y[c] = forward_y->p[j][i].s;
		    c++;
		}
	}
	CHECKCALL(cudaMemcpy(p->d_exp2pi_y_f, h_exp2pi_y_f, sizeof(CUCOMPLEX) * N_y * sRy, cudaMemcpyHostToDevice));
	CHECKCALL(cudaMemcpy(p->d_exp2pi_y_b, h_exp2pi_y_b, sizeof(CUCOMPLEX) * N_y * sRy, cudaMemcpyHostToDevice));
	CHECKCALL(cudaMemcpy(p->d_base_y, h_base_y, sizeof(int) * N_y * sRy, cudaMemcpyHostToDevice));
	CHECKCALL(cudaMemcpy(p->d_step_y, h_step_y, sizeof(int) * N_y * sRy, cudaMemcpyHostToDevice));
	
	
	CHECKCALL(cudaFreeHost(h_exp2pi_x_f));
	CHECKCALL(cudaFreeHost(h_exp2pi_y_f));
	CHECKCALL(cudaFreeHost(h_exp2pi_x_b));
	CHECKCALL(cudaFreeHost(h_exp2pi_y_b));
	CHECKCALL(cudaFreeHost(h_base_x));
	CHECKCALL(cudaFreeHost(h_base_y));
	CHECKCALL(cudaFreeHost(h_step_x));
	CHECKCALL(cudaFreeHost(h_step_y));
	// DONE making parts needed for future FFTs
	
	// temporary workspaces
	CHECKCALL(cudaMallocHost(&(p->h_temp), sCxy));
	CHECKCALL(cudaMalloc(&(p->d_A), sCxy));
	CHECKCALL(cudaMalloc(&(p->d_B), sCxy));


	// 2D arrays, 1st dimmension is for layer
	p->d_sx_q = new CUCOMPLEX*[nz];
	p->d_sy_q = new CUCOMPLEX*[nz];
	p->d_sz_q = new CUCOMPLEX*[nz];

	for(int i=0; i<nz; i++)
	{
		CHECKCALL(cudaMalloc(&(p->d_sx_q[i]), sCxy));
		CHECKCALL(cudaMalloc(&(p->d_sy_q[i]), sCxy));
		CHECKCALL(cudaMalloc(&(p->d_sz_q[i]), sCxy));
	}
	CHECKCALL(cudaMalloc(&(p->d_hA_q), sCxy));
	
	// make room for FT'd interaction matrices
	p->d_GammaXX = new CUCOMPLEX*[nz];
	p->d_GammaXY = new CUCOMPLEX*[nz];
	p->d_GammaXZ = new CUCOMPLEX*[nz];

	p->d_GammaYY = new CUCOMPLEX*[nz];
	p->d_GammaYZ = new CUCOMPLEX*[nz];
	
	p->d_GammaZZ = new CUCOMPLEX*[nz];

	for(int i=0; i<nz; i++)
	{
		CHECKCALL(cudaMalloc(&(p->d_GammaXX[i]), sCxy));
		CHECKCALL(cudaMalloc(&(p->d_GammaXY[i]), sCxy));
		CHECKCALL(cudaMalloc(&(p->d_GammaXZ[i]), sCxy));
	
		CHECKCALL(cudaMalloc(&(p->d_GammaYY[i]), sCxy));
		CHECKCALL(cudaMalloc(&(p->d_GammaYZ[i]), sCxy));
	
		CHECKCALL(cudaMalloc(&(p->d_GammaZZ[i]), sCxy));
	}
	
	
	CHECKCALL(cudaMalloc(&(p->d_output),sRxy));
	
	// now we will work on loading all the interaction matrices
	// onto the GPU and fourier transforming them
	struct {
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

	for(int k=0; k<6; k++) //XX XY XZ   YY YZ   ZZ
	{
		for(int j=0; j<nz; j++)
		{
			for(int c=0; c<nxy; c++)
			{
			    p->h_temp[c] = MAKECOMPLEX(sd[k].h[j*nxy + c], 0);
			}

			CHECKCALL(cudaMemcpy(p->d_A, p->h_temp, sizeof(CUCOMPLEX)*nxy, cudaMemcpyHostToDevice));
			
			fourier2D(N_x, N_y, p->Rx, p->Ry, 
				  p->d_exp2pi_x_f, p->d_exp2pi_y_f, 
				  p->d_base_x, p->d_base_y, 
				  p->d_step_x, p->d_step_y,
				  p->d_B, p->d_A);
// 				  sd[k].d[j], p->d_A);

			// going to prescale the data into d_GammaAB:
			d_scaleC(N_x, N_y, sd[k].d[j], p->d_B, 1.0/((double)(nxy)));
// 			d_scaleC(N_x, N_y, sd[k].d[j], p->d_B, 1.0);
		}
	}

// 	printf("\n");
// 	for(int k=0; k<6; k++) //XX XY XZ   YY YZ   ZZ
// 	{
// 	    CHECKCALL(cudaMemcpy(p->h_temp, sd[k].d[0], sCxy, cudaMemcpyDeviceToHost));
	    
// 		for(int c=0; c<N_x*N_y; c++)
// 		{
// 			printf("%g\n", p->h_temp[c].x);
// 		}
		
// 		printf("[");
// 		
// 		for(int j=0; j<N_y; j++)
// 		{
// 			for(int i=0; i<N_x; i++)
// 			{
// 				printf("%g + %gi  ", p->h_temp[i+j*N_x].x, p->h_temp[i+j*N_x].y);
// 			}
// 			if(j+1 < N_y)
// 				printf("; ");
// 		}
// 	    printf("]\n");
// 	}

	free_plan(forward_x);
	free_plan(forward_y);
	free_plan(backward_x);
	free_plan(backward_y);
	
	return p;
}

void free_JM_LONGRANGE_PLAN(JM_LONGRANGE_PLAN* p)
{
	const int N_z = p->N_z;
	const int nz = N_z; // * 2 - 1;
	
	CHECKCALL(cudaFree(p->d_exp2pi_x_f));
	CHECKCALL(cudaFree(p->d_exp2pi_y_f));
	CHECKCALL(cudaFree(p->d_exp2pi_x_b));
	CHECKCALL(cudaFree(p->d_exp2pi_y_b));

	CHECKCALL(cudaFree(p->d_output));
	
	CHECKCALL(cudaFreeHost(p->h_temp));
	
	for(int z=0; z<N_z; z++)
	{
		CHECKCALL(cudaFree(p->d_sx_q[z]));
		CHECKCALL(cudaFree(p->d_sy_q[z]));
		CHECKCALL(cudaFree(p->d_sz_q[z]));
	}
	CHECKCALL(cudaFree(p->d_hA_q));
	
	delete [] p->d_sx_q;
	delete [] p->d_sy_q;
	delete [] p->d_sz_q;

	for(int z=0; z<nz; z++)
	{
		CHECKCALL(cudaFree(p->d_GammaXX[z]));
		CHECKCALL(cudaFree(p->d_GammaXY[z]));
		CHECKCALL(cudaFree(p->d_GammaXZ[z]));
		
		CHECKCALL(cudaFree(p->d_GammaYY[z]));
		CHECKCALL(cudaFree(p->d_GammaYZ[z]));
		
		CHECKCALL(cudaFree(p->d_GammaZZ[z]));
	}
	
	delete [] p->d_GammaXX;
	delete [] p->d_GammaXY;
	delete [] p->d_GammaXZ;
	
	delete [] p->d_GammaYY;
	delete [] p->d_GammaYZ;
	
	delete [] p->d_GammaZZ;
	
	CHECKCALL(cudaFree(p->d_A));
	CHECKCALL(cudaFree(p->d_B));

	delete p;
}

__global__ void convolveSum(const int N_x, const int N_y, CUCOMPLEX* d_dest,  CUCOMPLEX* d_A,  CUCOMPLEX* d_B, double sign)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;
	
	const int idx = i + y * N_x;

#ifdef BOUND_CHECKS
	if(idx >= N_x * N_y) return;
#endif
	
	d_dest[idx] = cuCadd(d_dest[idx], cuCmul(d_A[idx], cuCmul(MAKECOMPLEX(sign,0), d_B[idx])));
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


	const int _b = i + y * N_x;
	const int _a = _b + layer * N_x * N_y;
#ifdef BOUND_CHECKS
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
	const int nxy = N_x * N_y;
	#ifdef SMART_SCHEDULE
	//different thread schedules for different access patterns
	dim3 blocks(N_x);
	dim3 threads(N_y);
	#else
	const int _blocksx = N_x / 32 + 1;
	const int _blocksy = N_y / 32 + 1;
	dim3 blocks(_blocksx, _blocksy);
	dim3 threads(32,32);
	#endif	

	CUCOMPLEX* d_src  = p->d_A; //local vars for swapping workspace
	CUCOMPLEX* d_dest = p->d_B;
	
	// FT the spins
	struct {
	    const double*	d_s_r;
	    CUCOMPLEX** 	d_s_q;
	    double*     	d_h_r;
	} sd[] = { //sd = static data
	    {d_sx, p->d_sx_q, d_hx},
	    {d_sy, p->d_sy_q, d_hy},
	    {d_sz, p->d_sz_q, d_hz}
	};

	for(int k=0; k<3; k++) // x y z
	{
		const double* d_s_r = sd[k].d_s_r;
		
		for(int z=0; z<N_z; z++)
		{
			d_src  = p->d_A;
			d_dest = p->d_B;

			//destination
			CUCOMPLEX* d_s_q = sd[k].d_s_q[z];

			d_r2c(N_x, N_y, d_dest, &(d_s_r[z*N_x*N_y]));

			fourier2D(N_x, N_y, p->Rx, p->Ry, 
						p->d_exp2pi_x_f, p->d_exp2pi_y_f, 
						p->d_base_x, p->d_base_y, 
						p->d_step_x, p->d_step_y,
						d_s_q, d_dest);
		}
	}

	// OK! Now we have all the spins FT'd and the interaction matrix ready.
	// We will now convolve the signals into hq
	

	// Nov 9/2011. Negative offsets are the same as positive offsets except tensors with odd number
	// of Zs are negated (XZ, YZ, not ZZ)
	for(int targetLayer=0; targetLayer<N_z; targetLayer++)
	{
	    for(int c=0; c<3; c++) //c = 0,1,2: X,Y,Z
	    {
		setC<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, 0, 0);
		KCHECK;
		
		for(int sourceLayer=0; sourceLayer<N_z; sourceLayer++)
		{
		    //const int offset = (sourceLayer - targetLayer + N_z - 1);
		    int offset = sourceLayer - targetLayer;
		    double sign = 1;
		    if(offset < 0)
		    {
			offset = -offset;
			sign = -1;
		    }

		    switch(c)
		    {
		    case 0:
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sx_q[sourceLayer], p->d_GammaXX[offset],    1);
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sy_q[sourceLayer], p->d_GammaXY[offset],    1);
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sz_q[sourceLayer], p->d_GammaXZ[offset], sign);
			break;
		    case 1:
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sx_q[sourceLayer], p->d_GammaXY[offset],    1);
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sy_q[sourceLayer], p->d_GammaYY[offset],    1);
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sz_q[sourceLayer], p->d_GammaYZ[offset], sign);
			break;
		    case 2:
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sx_q[sourceLayer], p->d_GammaXZ[offset], sign);
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sy_q[sourceLayer], p->d_GammaYZ[offset], sign);
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sz_q[sourceLayer], p->d_GammaZZ[offset],    1);
		    }
		    KCHECK
		}

		// h(q) now calculated, iFT it
		double* d_hxyz = sd[c].d_h_r; // this is where the result will go

		d_src = p->d_A;
		d_dest = p->d_B;
		
		fourier2D(N_x, N_y, p->Rx, p->Ry, 
			p->d_exp2pi_x_b, p->d_exp2pi_y_b, 
			p->d_base_x, p->d_base_y, 
			p->d_step_x, p->d_step_y,
			d_src, p->d_hA_q);
					
		//real space of iFFT in d_src, need to chop off the (hopefully) zero imag part
		getRPart<<<blocks, threads>>>(N_x, N_y, p->d_output, d_src);
		KCHECK;
			
		setLayer<<<blocks, threads>>>(N_x, N_y, targetLayer, d_hxyz, p->d_output);
		KCHECK;
	    }
	}
	
	//holy crap, we're done.
}

