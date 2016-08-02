#include <complex>
#include <iostream>
#include <vector>
using namespace std;
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cuComplex.h>
#include <stdio.h>

#include "longrange.h"

#include "../core_cuda/spinsystem.hpp"

#define JM_FORWARD 1
#define JM_BACKWARD 0

#define REAL float
//#define CUCOMPLEX cuDoubleComplex
#define CUCOMPLEX cuFloatComplex
//#define MAKECOMPLEX(a,b) make_cuDoubleComplex(a,b) 
#define MAKECOMPLEX(a,b) make_cuFloatComplex(a,b) 

// #define SMART_SCHEDULE

#ifdef SMART_SCHEDULE
  #define IDX_PATT(a, b) \

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
	{\
		printf("(%s:%i) %s\n",	__FILE__, __LINE__-1, cudaGetErrorString(i));\
		exit(-1);\
	}\
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








	
typedef struct fft_plan_phase
{
	int depth;
	int num_plan;
	int dest_count;
	
	int* h_src;
	int* h_dest;
	float* h_W_angles;

	int* d_src;
	int* d_dest;
	float* d_W_angles;
}fft_plan_phase;
	
//mixed radix plan
typedef struct fft_plan
{
	fft_plan_phase* phase;
	int num_phases;
	
	int n;
	int refcount;
} fft_plan;




static fft_plan** fft_plans = 0;
static int num_fft_plans;
static int size_fft_plans;


__global__ void getRPart32(const int N_x, const int N_y, REAL* d_dest,  CUCOMPLEX* d_src)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;

	const int idx = i + y * N_x;

	d_dest[idx] = d_src[idx].x;
}
__global__ void getIPart32(const int N_x, const int N_y, REAL* d_dest,  CUCOMPLEX* d_src)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;

	const int idx = i + y * N_x;

	d_dest[idx] = d_src[idx].y;
}


__global__ void scaleC32(const int N_x, const int N_y, CUCOMPLEX* d, CUCOMPLEX* s,  float v)
{
	IDX_PATT(x, y);
	
	if(x >= N_x || y >= N_y)
		return;
	
	const int idx = x + y * N_x;
	
//	d[idx] = cuCmulf(MAKECOMPLEX(v,0), s[idx]);
	d[idx].x = v * s[idx].x;
	d[idx].y = v * s[idx].y;
}
static void d_scaleC32(const int nx, const int ny, CUCOMPLEX* d_dest, CUCOMPLEX* d_src, float scale)
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
	
	scaleC32<<<blocks, threads>>>(nx, ny, d_dest, d_src, scale);
	KCHECK;
}




__global__ void setC32(const int N_x, const int N_y, CUCOMPLEX* v,  float R, float I)
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

	fft_plan* plan_x;
	fft_plan* plan_y;
	
	REAL* d_output;
	
	CUCOMPLEX* h_temp; 
	
	// 2D arrays, 1st dimmension is for layer
	CUCOMPLEX** d_sx_q;
	CUCOMPLEX** d_sy_q;
	CUCOMPLEX** d_sz_q;
	
	CUCOMPLEX*	d_hA_q;
	
	// 2D arrays, 1st dimmension is for interlayer offset
	CUCOMPLEX** d_GammaXX;
	CUCOMPLEX** d_GammaXY;
	CUCOMPLEX** d_GammaXZ;
	
	CUCOMPLEX** d_GammaYY;
	CUCOMPLEX** d_GammaYZ;
	
	CUCOMPLEX** d_GammaZZ;
}JM_LONGRANGE_PLAN;


static void add_plan(fft_plan* p)
{
	if(num_fft_plans == size_fft_plans)
	{
		size_fft_plans *= 2;
		fft_plans = (fft_plan**) realloc(fft_plans, sizeof(fft_plan) * size_fft_plans);
	}
	
	fft_plans[num_fft_plans] = p;
	num_fft_plans++;
}

static void free_plan(fft_plan* p)
{
	p->refcount--;
	
	if(p->refcount == 0)
	{
		for(int i=0; i<num_fft_plans; i++)
		{
			if(fft_plans[i] == p)
			{
				fft_plans[i] = 0;
			}
		}
		
		for(int i=0; i<p->num_phases; i++)
		{
			CHECKCALL(cudaFree(p->phase[i].d_src));
			CHECKCALL(cudaFree(p->phase[i].d_dest));
			CHECKCALL(cudaFree(p->phase[i].d_W_angles));
		}
		
		delete [] p->phase;
		free(p);
	}
}

#define ll_call(in,out) \
		if(lua_pcall(L, in, out, 0)) \
		{ \
			fprintf(stderr, "%s\n", lua_tostring(L, -1)); \
			lua_close(L); \
			return 0; \
		} 
		
static fft_plan* make_plan(int n)
{
	if(fft_plans == 0)
	{
		fft_plans = (fft_plan**) malloc(sizeof(fft_plan) * 32);
		num_fft_plans = 0;
		size_fft_plans = 32;
	}

	for(int i=0; i<num_fft_plans; i++)
	{
		if(fft_plans[i] && fft_plans[i]->n == n)
		{
			fft_plans[i]->refcount++;
			return fft_plans[i];
		}
	}
	
	lua_State* L = lua_open();
	luaL_openlibs(L);

	lua_newtable(L);
	for(int i=1; i<=n; i++)
	{
		lua_pushinteger(L, i);
		lua_pushinteger(L, i);
		lua_settable(L, -3);
	}
	lua_setglobal(L, "indices");
	
	if(luaL_dostring(L, __longrange))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		lua_close(L);
		return 0;
	}

	fft_plan* plan = new fft_plan;
	plan->n = n;

	lua_getglobal(L, "max_depth");
	int max_depth = lua_tointeger(L, -1);
	lua_pop(L, 1);

	plan->phase = new fft_plan_phase[max_depth];
	plan->num_phases = max_depth;
	
// 	printf("max_depth %i\n", max_depth);
	for(int i=0; i<max_depth; i++)
	{
		plan->phase[i].depth = i;
		lua_getglobal(L, "get_num_plan");
		lua_pushinteger(L, i+1);
		ll_call(1,1); 
		plan->phase[i].num_plan = lua_tointeger(L, -1);
		lua_pop(L, lua_gettop(L));
		
		lua_getglobal(L, "get_dest_count");
		lua_pushinteger(L, i+1);
		ll_call(1,1); 
		plan->phase[i].dest_count = lua_tointeger(L, -1);
		lua_pop(L, lua_gettop(L));
		
		const int np = plan->phase[i].num_plan;
		const int dc = plan->phase[i].dest_count;
		
		const int sz_i = dc * np * sizeof(int);
		const int sz_d = dc * np * sizeof(float);
		
		plan->phase[i].h_src      = (int*   )malloc(sz_i);
		plan->phase[i].h_dest     = (int*   )malloc(sz_i);
		plan->phase[i].h_W_angles = (float*)malloc(sz_d);

		CHECKCALL(malloc_device(&(plan->phase[i].d_src),  sz_i));
		CHECKCALL(malloc_device(&(plan->phase[i].d_dest), sz_i));
		CHECKCALL(malloc_device(&(plan->phase[i].d_W_angles), sz_d));

		
		int*    src = plan->phase[i].h_src;
		int*    dst = plan->phase[i].h_dest;
		float* wan = plan->phase[i].h_W_angles;
		
// 		printf("np: %i\n", np);
		for(int j=0; j<np; j++)
		{
			lua_getglobal(L, "get_plan");
			lua_pushinteger(L, i+1);
			lua_pushinteger(L, j+1);
			ll_call(2,3*dc); 
			
			for(int q=0; q<dc; q++)
			{
				src[j*dc+q] = lua_tointeger(L, q+1)-1;
// 				printf("%i ", src[j*dc+q]);
			}
			for(int q=0; q<dc; q++)
			{
				wan[j*dc+q] = -2.0*3.14159265358979* lua_tonumber(L, q+dc+1);
// 				printf("%g ", wan[j*dc+q]);
			}
			for(int q=0; q<dc; q++)
			{
				dst[j*dc+q] = lua_tointeger(L, q+dc*2+1)-1;
// 				printf("%i ", dst[j*dc+q]);
			}
// 			printf("\n");
			lua_pop(L, lua_gettop(L));
		}
		
		// move plan over to device and delete it here
		CHECKCALL(cudaMemcpy(
			plan->phase[i].d_src, 
			plan->phase[i].h_src, sz_i, cudaMemcpyHostToDevice));
		CHECKCALL(cudaMemcpy(
			plan->phase[i].d_dest, 
			plan->phase[i].h_dest, sz_i, cudaMemcpyHostToDevice));
		CHECKCALL(cudaMemcpy(
			plan->phase[i].d_W_angles, 
			plan->phase[i].h_W_angles, sz_d, cudaMemcpyHostToDevice));

		free(plan->phase[i].h_src);
		free(plan->phase[i].h_dest);
		free(plan->phase[i].h_W_angles);
	}

	add_plan(plan);
	return make_plan(n);
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
	
//	printf("%i %i %p %p\n", nx, ny, d_dest, d_src);
	__r2c<<<blocks, threads>>>(nx, ny, d_dest, d_src);
	KCHECK;
}


__device__ int d_sizeof_s_fft_iteration(int R)
{
	return sizeof(float) * sizeof(int) * (R+1);
}


template <int direction, int dest_count>
__global__ void Fourier_2D_x(const int nx, const int ny, 
			int* d_src, int* d_dest, float* d_W_angles,  
			CUCOMPLEX* dest, CUCOMPLEX* src)
{
	const int base = threadIdx.x * dest_count;
	const int y = blockIdx.x; 
	float Wreal, Wimag;
	CUCOMPLEX res;

	// fetch sources
	CUCOMPLEX s[dest_count];
// 	CUCOMPLEX d[dest_count];
#pragma unroll
	for(int i=0; i<dest_count; i++)
	{
		s[i] = src[y*nx+d_src[base+i]];
	}

	for(int i=0; i<dest_count; i++)
	{
		if(direction == 1) //forward
			sincos( d_W_angles[base+i] , &Wimag, &Wreal);
		else			   //backward
			sincos(-d_W_angles[base+i] , &Wimag, &Wreal);
		
		CUCOMPLEX  W	 = MAKECOMPLEX(Wreal, Wimag);
		CUCOMPLEX Wi	 = W;
		
		res = cuCaddf(s[0], cuCmulf(W, s[1]));
#pragma unroll
		for(int j=2; j<dest_count; j++)
		{
			Wi = cuCmulf(Wi, W);
			res = cuCaddf(res, cuCmulf(Wi, s[j]));
		}
		
		dest[y*nx+d_dest[base+i]] = res;
		// cache dest
// 		d[i] = res;
	}
	
// #pragma unroll
// 	for(int i=0; i<dest_count; i++)
// 	{
// 		// write to dest
// 		dest[y*nx+d_dest[base+i]] = d[i];
// 	}
}

// #define TESTING

template<int direction>
static  void fourier2D_x_element(
		fft_plan* plan, int nx, const int ny,
		CUCOMPLEX* d_dest, CUCOMPLEX* d_src)
{
	dim3 blocks(ny);

#ifdef TESTING
	
	CUCOMPLEX* h_data;
    CHECKCALL(malloc_host(&h_data, sizeof(CUCOMPLEX) * nx*ny));

	for(int i=0; i<ny; i++)
	{
		h_data[i*nx+0] = MAKECOMPLEX(1,0);
		h_data[i*nx+1] = MAKECOMPLEX(2,0);
		h_data[i*nx+2] = MAKECOMPLEX(1,0);
		h_data[i*nx+3] = MAKECOMPLEX(1,0);
		h_data[i*nx+4] = MAKECOMPLEX(1,0);
		h_data[i*nx+5] = MAKECOMPLEX(3,0);
		h_data[i*nx+6] = MAKECOMPLEX(3,0);
		h_data[i*nx+7] = MAKECOMPLEX(4,0);
	}
	//h_data[8] = MAKECOMPLEX(5,0);
	
	CHECKCALL(cudaMemcpy(d_src, h_data, sizeof(CUCOMPLEX)*nx*ny, cudaMemcpyHostToDevice));
#endif	
	
	CUCOMPLEX* T;
	for(int phase=plan->num_phases-1; phase>=0; phase--)
// 	int phase=plan->num_phases-1;
	{
		struct fft_plan_phase& p = plan->phase[phase];
		
		dim3 threads(p.num_plan);
		
// 		template <int direction, int dest_count>
// __global__ void Fourier_2D_x(const int nx, const int ny, 
// 			int* d_src, int* d_dest, float* d_W_angles,  
// 			CUCOMPLEX* dest, CUCOMPLEX* src)

#define FFF(dc) case dc: \
	Fourier_2D_x<direction,dc><<<blocks, threads>>>(nx, ny, \
		p.d_src, p.d_dest, p.d_W_angles, d_dest, d_src); \
	break
		switch(p.dest_count)
		{
			FFF( 2);
			FFF( 3);
			FFF( 4);
			FFF( 5);
			FFF( 6);
			FFF( 7);
			FFF( 8);
			FFF( 9);
			FFF(10);
			FFF(11);
			FFF(12);
			FFF(13);
			FFF(14);
			FFF(15);
			FFF(16);
			FFF(17);
			default:
				fprintf(stderr, "Spurious dest_count[%i] (%s:%i)\n", p.dest_count, __FILE__, __LINE__);
		}
		KCHECK;
#undef FFF
		T = d_src;
		d_src = d_dest;
		d_dest = T;
	}
	
	
	#ifdef TESTING

	printf("Goal:\n");
	printf("   16.0000 +  0.0000i\n");
	printf("    1.4142 +  4.8284i\n");
	printf("   -2.0000 +  0.0000i\n");
	printf("   -1.4142 +  0.8284i\n");
	printf("   -4.0000 +  0.0000i\n");
	printf("   -1.4142 -  0.8284i\n");
	printf("   -2.0000 -  0.0000i\n");
	printf("    1.4142 -  4.8284i\n");

// 	printf("   9.0000 + 0.0000i\n");
// 	printf("   1.5000 + 0.8660i\n");
// 	printf("  -1.5000 + 0.8660i\n");
// 	printf("  -3.0000 + 0.0000i\n");
// 	printf("  -1.5000 - 0.8660i\n");
// 	printf("   1.5000 - 0.8660i\n");

	for(int q=1; q>=0; q--)
	{
		if(q == 0)
		{
			CHECKCALL(cudaMemcpy(h_data, d_dest, sizeof(CUCOMPLEX)*nx*ny, cudaMemcpyDeviceToHost));
		}
		else
		{
			CHECKCALL(cudaMemcpy(h_data, d_src, sizeof(CUCOMPLEX)*nx*ny, cudaMemcpyDeviceToHost));
		}

		printf("\nres (%i):\n", q);
		for(int i=0; i<nx; i++)
		{
			printf("%i)  % -.4f %s % -.4fi\n", i, h_data[nx+i].x, h_data[nx+i].y<0?"-":"+", fabs(h_data[nx+i].y));
		}
	}
	exit(-1);
#endif
}

// template<int direction>
// static void fourier2D_x(
// 		fft_plan* plan, int nx, const int ny,
// 		CUCOMPLEX* d_dest, CUCOMPLEX* d_src)
// {
// 
// 
// #ifdef TESTING
// 	
// 	CUCOMPLEX* h_data;
//     CHECKCALL(malloc_host(&h_data, sizeof(CUCOMPLEX) * nx));
// 
// 	h_data[0] = MAKECOMPLEX(1,0);
// 	h_data[1] = MAKECOMPLEX(2,0);
// 	h_data[2] = MAKECOMPLEX(1,0);
// 	h_data[3] = MAKECOMPLEX(1,0);
// 	h_data[4] = MAKECOMPLEX(1,0);
// 	h_data[5] = MAKECOMPLEX(3,0);
// 	//h_data[6] = MAKECOMPLEX(3,0);
// 	//h_data[7] = MAKECOMPLEX(4,0);
// 	//h_data[8] = MAKECOMPLEX(5,0);
// 	
// 	CHECKCALL(cudaMemcpy(d_src, h_data, sizeof(CUCOMPLEX)*nx, cudaMemcpyHostToDevice));
// 
// 	printf("Passes: %i\n", passes);
// #endif
// 
// 	fourier2D_x_element(plan, nx, ny, d_dest, d_src); // fft now in d_src
// 
// // // 	Fourier_2D_x<direction><<<blocks, threads>>>(nx, ny, plan->d_p, plan->d_radix, nradix, d_dest, d_src, passes);
// 	/*
// 	if(!(nradix & 0x1)) //then even number of operations, need to copy sol'n to d_dest
// 	{
// 		CHECKCALL(cudaMemcpy(d_dest, d_src, sizeof(CUCOMPLEX)*nx*ny, cudaMemcpyDeviceToDevice));
// 		}*/
// //	printf("fourier2D (%s;%i)\n", __FILE__, __LINE__);
// 
// #ifdef TESTING
// 
// 	printf("Goal:\n");
// 	printf("   9.0000 + 0.0000i\n");
// 	printf("   1.5000 + 0.8660i\n");
// 	printf("  -1.5000 + 0.8660i\n");
// 	printf("  -3.0000 + 0.0000i\n");
// 	printf("  -1.5000 - 0.8660i\n");
// 	printf("   1.5000 - 0.8660i\n");
// 
// 	for(int q=0; q<2; q++)
// 	{
// 		if(q == 0)
// 		{
// 			CHECKCALL(cudaMemcpy(h_data, d_dest, sizeof(CUCOMPLEX)*nx, cudaMemcpyDeviceToHost));
// 		}
// 		else
// 		{
// 			CHECKCALL(cudaMemcpy(h_data, d_src, sizeof(CUCOMPLEX)*nx, cudaMemcpyDeviceToHost));
// 		}
// 
// 		printf("\nres (%i):\n", q);
// 		for(int i=0; i<nx; i++)
// 		{
// 			printf("%i)  %g%+gi\n", i, h_data[i].x, h_data[i].y);
// 		}
// 	}
// 	exit(-1);
// #endif
// }

__global__ void transposeSimple(int nx, int ny, CUCOMPLEX *d_dest, CUCOMPLEX *d_src)
{
	IDX_PATT(x, y);

	if(x >= nx || y >= ny)
		return;
	
	d_dest[x*ny + y] = d_src[y*nx + x];
}

template<int direction>
static void fourier2D_Transposed(
		fft_plan* planx, fft_plan* plany, const int nx, const int ny,
		CUCOMPLEX* d_dest, CUCOMPLEX* d_src)
{	
	const int npx = planx->num_phases;
	const int npy = plany->num_phases;

	fourier2D_x_element<direction>(planx, nx, ny, d_dest, d_src);
	
	//if npx is odd then results are in d_dest
	
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

	if(npx % 2) //then odd, fft res in d_dest
	{
		transposeSimple<<<blocks, threads>>>(nx, ny, d_src, d_dest);
		fourier2D_x_element<direction>(plany, ny, nx, d_dest, d_src);
	}
	else
	{
		transposeSimple<<<blocks, threads>>>(nx, ny, d_dest, d_src);
		fourier2D_x_element<direction>(plany, ny, nx, d_src, d_dest);
	}
	
	if((npx + npy) % 2)
	{
		CHECKCALL(cudaMemcpy(d_dest, d_src, sizeof(CUCOMPLEX)*nx*ny, cudaMemcpyDeviceToDevice));
	}
}


int JM_LONGRANGE_PLAN_ws_size(int nx, int ny, int /*nz*/)
{
	return sizeof(CUCOMPLEX) * nx*ny;
}

JM_LONGRANGE_PLAN* make_JM_LONGRANGE_PLAN(int N_x, int N_y, int N_z, 
	double* GammaXX, double* GammaXY, double* GammaXZ,
					 double* GammaYY, double* GammaYZ,
									  double* GammaZZ, void* ws_d_A, void* ws_d_B)
{
	const int nz   = N_z;
	const int nxy = N_x * N_y;
	const int sRxy = sizeof(REAL) * nxy;
	const int sCxy = sizeof(CUCOMPLEX) * nxy;

	JM_LONGRANGE_PLAN* p = new JM_LONGRANGE_PLAN;
	
	p->N_x = N_x;
	p->N_y = N_y;
	p->N_z = N_z;
	
	p->plan_x = make_plan(N_x);
	p->plan_y = make_plan(N_y);

	// temporary workspaces (host)
	CHECKCALL(malloc_host(&(p->h_temp), sCxy));

	CUCOMPLEX* d_A = (CUCOMPLEX*)ws_d_A;
	CUCOMPLEX* d_B = (CUCOMPLEX*)ws_d_B;
	
	// 2D arrays, 1st dimmension is for layer
	p->d_sx_q = new CUCOMPLEX*[nz];
	p->d_sy_q = new CUCOMPLEX*[nz];
	p->d_sz_q = new CUCOMPLEX*[nz];

	for(int i=0; i<nz; i++)
	{
		CHECKCALL(malloc_device(&(p->d_sx_q[i]), sCxy));
		CHECKCALL(malloc_device(&(p->d_sy_q[i]), sCxy));
		CHECKCALL(malloc_device(&(p->d_sz_q[i]), sCxy));
	}
	CHECKCALL(malloc_device(&(p->d_hA_q), sCxy));
	
	// make room for FT'd interaction matrices
	p->d_GammaXX = new CUCOMPLEX*[nz];
	p->d_GammaXY = new CUCOMPLEX*[nz];
	p->d_GammaXZ = new CUCOMPLEX*[nz];

	p->d_GammaYY = new CUCOMPLEX*[nz];
	p->d_GammaYZ = new CUCOMPLEX*[nz];
	
	p->d_GammaZZ = new CUCOMPLEX*[nz];

	for(int i=0; i<nz; i++)
	{
		CHECKCALL(malloc_device(&(p->d_GammaXX[i]), sCxy));
		CHECKCALL(malloc_device(&(p->d_GammaXY[i]), sCxy));
		CHECKCALL(malloc_device(&(p->d_GammaXZ[i]), sCxy));
	
		CHECKCALL(malloc_device(&(p->d_GammaYY[i]), sCxy));
		CHECKCALL(malloc_device(&(p->d_GammaYZ[i]), sCxy));
	
		CHECKCALL(malloc_device(&(p->d_GammaZZ[i]), sCxy));
	}
	
	
	CHECKCALL(malloc_device(&(p->d_output),sRxy));
	
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

	for(int k=0; k<6; k++) //XX XY XZ	YY YZ	ZZ
	{
		for(int j=0; j<nz; j++)
		{
			for(int c=0; c<nxy; c++)
			{
				p->h_temp[c] = MAKECOMPLEX(sd[k].h[j*nxy + c], 0);
			}

			CHECKCALL(cudaMemcpy(d_A, p->h_temp, sizeof(CUCOMPLEX)*nxy, cudaMemcpyHostToDevice));
			
			fourier2D_Transposed<1>(p->plan_x, p->plan_y, N_x, N_y, d_B, d_A);
			KCHECK;
			
			// going to prescale the data into d_GammaAB:
			d_scaleC32(N_x, N_y, sd[k].d[j], d_B, 1.0/((float)(nxy)));
//			d_scaleC(N_x, N_y, sd[k].d[j], p->d_B, 1.0);
		}
	}

	return p;
}

void free_JM_LONGRANGE_PLAN(JM_LONGRANGE_PLAN* p)
{
	const int N_z = p->N_z;
	const int nz = N_z; // * 2 - 1;
	
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

	free_plan(p->plan_x);
	free_plan(p->plan_y);
	
	delete p;
}

__global__ void convolveSum(const int N_x, const int N_y, CUCOMPLEX* d_dest,  CUCOMPLEX* d_A,  CUCOMPLEX* d_B, float sign)
{
	IDX_PATT(i, y);
	
	if(i >= N_x || y >= N_y)
		return;
	
	const int idx = i + y * N_x;

#ifdef BOUND_CHECKS
	if(idx >= N_x * N_y) return;
#endif
	
	d_dest[idx] = cuCaddf(d_dest[idx], cuCmulf(d_A[idx], cuCmulf(MAKECOMPLEX(sign,0), d_B[idx])));
}


__global__ void getLayer32(const int nx, const int ny, const int layer, REAL* d_dest,	 const REAL* d_src)
{
	IDX_PATT(row, col);
	
	if(row >= nx || col >= ny)
		return;

	const int _a = col + row*nx;
	const int _b = _a + layer*nx*ny;
	
	d_dest[_a] = d_src[_b];
}

__global__ void setLayer32(const int N_x, const int N_y, const int layer, REAL* d_dest,  REAL* d_src)
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
				  const float* d_sx, const float* d_sy, const float* d_sz,
				  float* d_hx, float* d_hy, float* d_hz, void* ws_d_A, void* ws_d_B)
{
	const int N_x = p->N_x;
	const int N_y = p->N_y;
	const int N_z = p->N_z;
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

	CUCOMPLEX* d_A = (CUCOMPLEX*)ws_d_A;
	CUCOMPLEX* d_B = (CUCOMPLEX*)ws_d_B;
		
	CUCOMPLEX* d_src  = d_A; //local vars for swapping workspace
	CUCOMPLEX* d_dest = d_B;
	
	// FT the spins
	struct {
		const float*	d_s_r;
		CUCOMPLEX**		d_s_q;
		float*			d_h_r;
	} sd[] = { //sd = static data
		{d_sx, p->d_sx_q, d_hx},
		{d_sy, p->d_sy_q, d_hy},
		{d_sz, p->d_sz_q, d_hz}
	};

	for(int k=0; k<3; k++) // x y z
	{
		const float* d_s_r = sd[k].d_s_r;
		
		for(int z=0; z<N_z; z++)
		{
			d_src  = d_A;
			d_dest = d_B;

			//destination
			CUCOMPLEX* d_s_q = sd[k].d_s_q[z];

			d_r2c(N_x, N_y, d_dest, &(d_s_r[z*N_x*N_y]));

			fourier2D_Transposed<1>(p->plan_x, p->plan_y, N_x, N_y, d_s_q, d_dest);

			
//			fourier2D(N_x, N_y, p->Rx, p->Ry, 
//						p->d_exp2pi_x_f, p->d_exp2pi_y_f, 
//						p->d_base_x, p->d_base_y, 
//						p->d_step_x, p->d_step_y,
//						d_s_q, d_dest);
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
		setC32<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, 0, 0);
		KCHECK;
		
		for(int sourceLayer=0; sourceLayer<N_z; sourceLayer++)
		{
			//const int offset = (sourceLayer - targetLayer + N_z - 1);
			int offset = sourceLayer - targetLayer;
			float sign = 1;
			if(offset < 0)
			{
			offset = -offset;
			sign = -1;
			}

			switch(c)
			{
			case 0:
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sx_q[sourceLayer], p->d_GammaXX[offset],	   1);
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sy_q[sourceLayer], p->d_GammaXY[offset],	   1);
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sz_q[sourceLayer], p->d_GammaXZ[offset], sign);
			break;
			case 1:
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sx_q[sourceLayer], p->d_GammaXY[offset],	   1);
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sy_q[sourceLayer], p->d_GammaYY[offset],	   1);
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sz_q[sourceLayer], p->d_GammaYZ[offset], sign);
			break;
			case 2:
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sx_q[sourceLayer], p->d_GammaXZ[offset], sign);
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sy_q[sourceLayer], p->d_GammaYZ[offset], sign);
			convolveSum<<<blocks, threads>>>(N_x, N_y, p->d_hA_q, p->d_sz_q[sourceLayer], p->d_GammaZZ[offset],	   1);
			}
			KCHECK
		}

		// h(q) now calculated, iFT it
		float* d_hxyz = sd[c].d_h_r; // this is where the result will go

		d_src = d_A;
		d_dest = d_B;
		
		fourier2D_Transposed<-1>(p->plan_y, p->plan_x, N_y, N_x, d_src, p->d_hA_q);

		
//		fourier2D(N_x, N_y, p->Rx, p->Ry, 
//			p->d_exp2pi_x_b, p->d_exp2pi_y_b, 
//			p->d_base_x, p->d_base_y, 
//			p->d_step_x, p->d_step_y,
//			d_src, p->d_hA_q);
//					
		//real space of iFFT in d_src, need to chop off the (hopefully) zero imag part
		getRPart32<<<blocks, threads>>>(N_x, N_y, p->d_output, d_src);
		KCHECK;
			
		setLayer32<<<blocks, threads>>>(N_x, N_y, targetLayer, d_hxyz, p->d_output);
		KCHECK;
		}
	}
	
	//holy crap, we're done.
}

