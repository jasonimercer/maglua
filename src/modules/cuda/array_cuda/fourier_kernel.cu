#include <complex>
#include <iostream>
#include <vector>
#include "luabaseobject.h"

using namespace std;
#include <stdlib.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "fourier.h"
#include "memory.hpp"

#include "hd_helper_tfuncs.hpp"

#define FFT_FORWARD -1
#define FFT_BACKWARD 1



#define IDX_PATT(a, b) \
	const int a = blockDim.x * blockIdx.x + threadIdx.x; \
	const int b = blockDim.y * blockIdx.y + threadIdx.y;

#define IDX_PATT3(a, b, c) \
	const int a = blockDim.x * blockIdx.x + threadIdx.x; \
	const int b = blockDim.y * blockIdx.y + threadIdx.y; \
	const int c = blockDim.z * blockIdx.z + threadIdx.z;

	

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

	int* d_src;
	int* d_dest;
	void* d_W_angles;
}fft_plan_phase;
	
//mixed radix plan
typedef struct fft_plan
{
	fft_plan_phase* phase;
	int num_phases;
	
	int n;
	int refcount;
	int sizeofT;
} fft_plan;


typedef struct FFT_PLAN
{
	int nx, ny, nz;
	int direction;
	int dims;
	fft_plan* plans[3];
}FFT_PLAN;



static fft_plan** fft_plans = 0;
static int num_fft_plans;
static int size_fft_plans;




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
	if(!p) return;
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

template<typename T>
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
		if(fft_plans[i] && fft_plans[i]->n == n && fft_plans[i]->sizeofT == sizeof(T))
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
	
	if(luaL_dostring(L, __fourier))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		lua_close(L);
		return 0;
	}

	fft_plan* plan = new fft_plan;
	plan->n = n;
	plan->refcount = 0;
	plan->sizeofT = sizeof(T);

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
		const int sz_d = dc * np * sizeof(T);
		
		plan->phase[i].h_src      = (int*   )malloc(sz_i);
		plan->phase[i].h_dest     = (int*   )malloc(sz_i);
		T* h_W_angles = (T*)malloc(sz_d);

		CHECKCALL(malloc_device(&(plan->phase[i].d_src),  sz_i));
		CHECKCALL(malloc_device(&(plan->phase[i].d_dest), sz_i));
		CHECKCALL(malloc_device(&(plan->phase[i].d_W_angles), sz_d));
		
		int*    src = plan->phase[i].h_src;
		int*    dst = plan->phase[i].h_dest;
		T* wan = h_W_angles;
		
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
			h_W_angles, sz_d, cudaMemcpyHostToDevice));

		free(plan->phase[i].h_src);
		free(plan->phase[i].h_dest);
		free(h_W_angles);
// 		free(plan->phase[i].h_W_angles);
	}

	lua_close(L);
	add_plan(plan);
	return make_plan<T>(n);
}


template<typename T>
__device__ int d_sizeof_s_fft_iteration(int R)
{
	return sizeof(T) * sizeof(int) * (R+1);
}


template <typename CPLX, typename REAL, int direction, int dest_count>
__global__ void Fourier_3D_x(const int nx, const int ny, 
			int* d_src, int* d_dest, REAL* d_W_angles,  
			CPLX* dest, CPLX* src)
{
	const int base = threadIdx.x * dest_count;
	const int yoff = blockIdx.x * nx; 
	const int zoff = blockIdx.y * nx * ny;
	REAL Wreal, Wimag;
	CPLX res;

	// fetch sources
	CPLX s[dest_count];
#pragma unroll
	for(int i=0; i<dest_count; i++)
	{
		s[i] = src[zoff + yoff + d_src[base+i]];
	}

	for(int i=0; i<dest_count; i++)
	{
		if(direction ==-1) //forward
			sincosT<REAL>( d_W_angles[base+i] , &Wimag, &Wreal);
		else			   //backward
			sincosT<REAL>(-d_W_angles[base+i] , &Wimag, &Wreal);
		
		CPLX  W;
		W.x = Wreal;
		W.y = Wimag;
		CPLX Wi	 = W;
		
		
		//res = cuCadd(s[0], cuCmul(W, s[1]));
		res = W;
		times_equal<CPLX>(res, s[1]);
		plus_equal<CPLX>(res, s[0]);
		
#pragma unroll
		for(int j=2; j<dest_count; j++)
		{
			//Wi = cuCmul(Wi, W);
			times_equal<CPLX>(Wi, W);
			
			//res = cuCadd(res, cuCmul(Wi, s[j]));
			CPLX t = Wi;
			times_equal(t, s[j]);
			plus_equal(res, t);
		}
		
		dest[zoff + yoff + d_dest[base+i]] = res;
	}
}


template<typename CPLX, typename REAL, int direction>
static  void fourier3D_x(
		fft_plan* plan, const int nx, const int ny, const int nz,
		CPLX* d_dest, CPLX* d_src)
{
	CPLX* true_dest = d_dest;
	dim3 blocks(ny,nz);

	CPLX* T;
	for(int phase=plan->num_phases-1; phase>=0; phase--)
	{
		struct fft_plan_phase& p = plan->phase[phase];
		
		dim3 threads(p.num_plan);

		#define FFF(dc) \
			case dc: \
				Fourier_3D_x<CPLX,REAL,direction,dc><<<blocks, threads>>> \
				(nx, ny, p.d_src, p.d_dest, (REAL*)p.d_W_angles, d_dest, d_src);\
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
	
	if(d_src != true_dest) //d_src because of swap above
	{
		memcpy_d2d(true_dest, d_src, sizeof(CPLX)*nx*ny*nz);
	}
		
}






template <typename CPLX, int fixed_coord>
__global__ void _transpose3D(const int nx, const int ny, const int nz, CPLX *d_dest, CPLX *d_src)
{
	IDX_PATT3(x, y, z);

	if(x >= nx || y >= ny || z >= nz)
		return;
	
#define C3(_x,_y,_z,_nx,_ny,_nz)  _x + _y*_nx + _z*_nx*_ny
	if(fixed_coord == 0) //swap y, z
	{
		 d_dest[C3(x,z,y, nx,nz,ny)] = d_src[C3(x,y,z, nx,ny,nz)];
	}
	if(fixed_coord == 1) //swap x, z
	{
		 d_dest[C3(z,y,x, nz,ny,nx)] = d_src[C3(x,y,z, nx,ny,nz)];
	}
	if(fixed_coord == 2) //swap x, y
	{
		 d_dest[C3(y,x,z, ny,nx,nz)] = d_src[C3(x,y,z, nx,ny,nz)];
	}
#undef C3

// 	d_dest[x*ny + y] = d_src[y*nx + x];
}

template<typename CPLX, int fixed_coord>
void transpose3D(const int nx, const int ny, const int nz, CPLX* dest, CPLX* src)
{
	const int _blocksx = nx / 32 + 1;
	const int _blocksy = ny / 32 + 1;
	dim3 blocks(_blocksx, _blocksy, nz);
	dim3 threads(32,32,1);
	
	_transpose3D<CPLX,fixed_coord><<<blocks, threads>>>(nx,ny,nz,dest,src);
}



template <typename T>
static FFT_PLAN* make_FFT_PLAN_T(int direction, int fftdims, const int nx, const int ny, const int nz)
{
	FFT_PLAN* p = (FFT_PLAN*)malloc(sizeof(FFT_PLAN));
	p->nx = nx;
	p->ny = ny;
	p->nz = nz;
	p->dims = fftdims;
	p->direction = direction;
	
	if(p->dims < 0)
		p->dims = 0;
	if(p->dims > 2)
		p->dims = 2;
	
	p->plans[0] = 0;
	p->plans[1] = 0;
	p->plans[2] = 0;

	const int n[3] = {nx,ny,nz};
	
	for(int i=0; i<p->dims; i++)
	{
		p->plans[i] = make_plan<T>(n[i]);
	}
	
	return p;
}


FFT_PLAN* make_FFT_PLAN_double(int direction, int fftdims, const int nx, const int ny, const int nz)
{
	return make_FFT_PLAN_T<double>(direction, fftdims, nx, ny, nz);
}

FFT_PLAN* make_FFT_PLAN_float(int direction, int fftdims, const int nx, const int ny, const int nz)
{
	return make_FFT_PLAN_T<float>(direction, fftdims, nx, ny, nz);
}


// template<typename CPLX, typename REAL, int direction>
// static  void fourier3D_x(
// 		fft_plan* plan, const int nx, const int ny, const int nz,
// 		CPLX* d_dest, CPLX* d_src);



template <typename CPLX, typename REAL, int direction>
void execute_FFT_PLAN_T(FFT_PLAN* plan, CPLX* dest, CPLX* src, CPLX* ws)
{
	const int nx = plan->nx;
	const int ny = plan->ny;
	const int nz = plan->nz;
	
	switch(plan->dims)
	{
		case 1:
		memcpy_d2d(ws, src, sizeof(CPLX)*nx*ny*nz);
		fourier3D_x<CPLX,REAL,direction>(plan->plans[0], nx,ny,nz, dest, ws); //FFT in dest
		break;
		
		case 2:
		memcpy_d2d(dest, src, sizeof(CPLX)*nx*ny*nz);
		fourier3D_x<CPLX,REAL,direction>(plan->plans[0], nx,ny,nz, ws, dest); //FFT in ws
		transpose3D<CPLX,2>(nx,ny,nz, dest, ws); //FFT in dest
		fourier3D_x<CPLX,REAL,direction>(plan->plans[1], ny,nx,nz, ws, dest); //FFT in ws
		transpose3D<CPLX,2>(ny,nx,nz, dest, ws); //FFT in dest
		break;
		
		case 3:
		memcpy_d2d(ws, src, sizeof(CPLX)*nx*ny*nz);
		fourier3D_x<CPLX,REAL,direction>(plan->plans[0], nx,ny,nz, dest, ws);
		transpose3D<CPLX,2>(nx,ny,nz, ws, dest); //FFT in dest
		fourier3D_x<CPLX,REAL,direction>(plan->plans[1], ny,nx,nz, dest, ws); 
		// the following two lines could be combined into 1 if 
		// some kind of arbitrary coord rotate function existed
		transpose3D<CPLX,2>(ny,nx,nz, ws, dest);
		transpose3D<CPLX,1>(nx,ny,nz, dest, ws);
		fourier3D_x<CPLX,REAL,direction>(plan->plans[2], nz,ny,nx, ws, dest);
		transpose3D<CPLX,1>(nz,ny,nx, dest, ws);
		break;
	}
}

void execute_FFT_PLAN(FFT_PLAN* plan, cuFloatComplex* dest, cuFloatComplex* src, cuFloatComplex* ws)
{
	switch(plan->direction)
	{
		case -1:	execute_FFT_PLAN_T<cuFloatComplex,float,-1>(plan, dest, src, ws); break;
		case  1:	execute_FFT_PLAN_T<cuFloatComplex,float, 1>(plan, dest, src, ws); break;
		default:
			fprintf(stderr, "(%s:%i) unknown fft direction: %i\n", __FILE__, __LINE__, plan->direction);
	}
}

void execute_FFT_PLAN(FFT_PLAN* plan, cuDoubleComplex* dest, cuDoubleComplex* src, cuDoubleComplex* ws)
{
	switch(plan->direction)
	{
		case -1:	execute_FFT_PLAN_T<cuDoubleComplex,double,-1>(plan, dest, src, ws); break;
		case  1:	execute_FFT_PLAN_T<cuDoubleComplex,double, 1>(plan, dest, src, ws); break;
		default:
			fprintf(stderr, "(%s:%i) unknown fft direction: %i\n", __FILE__, __LINE__, plan->direction);
	}
}

void free_FFT_PLAN(FFT_PLAN* p)
{
	free_plan(p->plans[0]);
	free_plan(p->plans[1]);
	free_plan(p->plans[2]);
	free(p);
}

