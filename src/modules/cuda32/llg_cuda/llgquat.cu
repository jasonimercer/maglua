#include <cuda.h>
#include <cuda_runtime.h>
#include <math_functions.h>
#include <stdio.h>

#define CROSS(v, a, b) \
	v.x = a.y * b.z - a.z * b.y; \
	v.y = a.z * b.x - a.x * b.z; \
	v.z = a.x * b.y - a.y * b.x;
	
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


__device__ float4 qconjugate(const float4 q)
{
	float4 r;
	r.x = -1.0 * q.x;
	r.y = -1.0 * q.y;
	r.z = -1.0 * q.z;
	r.w =        q.w;
	return r;
}

__device__ float4 qmult(const float4 a, const float4 b)
{
	float4 ab;

	ab.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
	ab.x = a.w*b.x + b.w*a.x  + a.y*b.z - a.z*b.y;
	ab.y = a.w*b.y + b.w*a.y  + a.z*b.x - a.x*b.z;
	ab.z = a.w*b.z + b.w*a.z  + a.x*b.y - a.y*b.x;

	return ab;
}

// quat mult without calculating the W component
__device__ float4 qmultXYZ(const float4 a, const float4 b)
{
	float4 ab;

	ab.x = a.w*b.x + b.w*a.x  + a.y*b.z - a.z*b.y;
	ab.y = a.w*b.y + b.w*a.y  + a.z*b.x - a.x*b.z;
	ab.z = a.w*b.z + b.w*a.z  + a.x*b.y - a.y*b.x;
	ab.w = 0;

	return ab;
}

// _1 will compute rhs and store in ws
// dS    -g           a
// -- = ---- S X (H +---S X H)
// dt   1+aa         |S|
__global__ void llg_quat_apply_1_32(
//	const int nx, const int ny, const int offset,
	const int nxyz,
	float alpha,
	float* sx, float* sy, float* sz, float* sms,
	float* hx, float* hy, float* hz,
        float* htx, float* hty, float* htz,              // dm/dt thermal fields
	// ws1, ws2, ws3,     ws4
	float* wx, float* wy, float* wz, float* ww) 
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i >= nxyz)
		return;

	// subtracting thermal term from Heff in damping contribution
	const float hX = hx[i] - htx[i];
	const float hY = hy[i] - hty[i];
	const float hZ = hz[i] - htz[i];
	
	wx[i] = sy[i]*hZ - sz[i]*hY;
	wy[i] = sz[i]*hX - sx[i]*hZ;
	wz[i] = sx[i]*hY - sy[i]*hX;
	
	ww[i] = sms[i];
	if(ww[i] != 0)
	{
		ww[i] = FAST_DIV(alpha, ww[i]);
	}
	wx[i] *= ww[i];
	wy[i] *= ww[i];
	wz[i] *= ww[i];
	
	wx[i] += hx[i];
	wy[i] += hy[i];
	wz[i] += hz[i];

	ww[i] = sqrt(wx[i]*wx[i] + wy[i]*wy[i] + wz[i]*wz[i]);
	
	//(wx, wy, wz) = (a / |S|) S x H
	// ww = | (wx, wy, wz) |
}

// _2 will compute the rest
// dS    -g           a
// -- = ---- S X (H +---S X H)
// dt   1+aa         |S|
// the rhs vec and len is in ws
__global__ void llg_quat_apply_2_32(
//	const int nx, const int ny, const int offset,
	const int nxyz,
	float* ssx, float* ssy, float* ssz, // src
	float* wx, float* wy, float* wz, float* ww,
	float alpha, float gadt)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i >= nxyz)
		return;

	if(ww[i] == 0) // dst = src  in _3
	{
		return;
	}

// dS    -g           a
// -- = ---- S X (H +---S X H)
// dt   1+aa         |S|
	float4 qVec;
	qVec.x = ssx[i]; 
	qVec.y = ssy[i];
	qVec.z = ssz[i];
	qVec.w = 0;
			
	const float theta = ww[i] * gadt;

	float cost, sint;
	sincos(theta, &sint, &cost);
	const float ihlen = FAST_DIV(1, ww[i]);

	float4 qRot;
	qRot.x = sint * wx[i] * ihlen; 
	qRot.y = sint * wy[i] * ihlen;
	qRot.z = sint * wz[i] * ihlen;
	qRot.w = cost;

	//this is the rotation: qRes = qRot qVec qRot*
	float4 qRes = qmultXYZ(qmult(qRot, qVec), qconjugate(qRot));

	wx[i] = qRes.x;
	wy[i] = qRes.y;
	wz[i] = qRes.z;
	
	// (wx, wy, wz) = (qRot qVec) conj(qRot)
	// ww = unimportant
}

// _3 normalize
__global__ void llg_quat_apply_3_32(
//	const int nx, const int ny, const int offset,
	const int nxyz,
	float*  wx, float*  wy, float*  wz, float* ww,
	float* ssx, float* ssy, float* ssz, float* sms,
 	float* dsx, float* dsy, float* dsz, float* dms)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i >= nxyz)
		return;

	dms[i] = sms[i];
	
	if(ww[i] == 0)
	{
		dsx[i] = ssx[i];
		dsy[i] = ssy[i];
		dsz[i] = ssz[i];
		return;
	}
	
	//using ww as temp var. saves a reg?
	ww[i] = sqrt(wx[i]*wx[i] + wy[i]*wy[i] + wz[i]*wz[i]);
	
	if(ww[i] == 0)
	{
		dsx[i] = ssx[i];
		dsy[i] = ssy[i];
		dsz[i] = ssz[i];
		return;
	}
	
	
	ww[i] = sms[i] / ww[i];

	// (wx, wy, wz) = (qRot qVec) conj(qRot)
	// ww = | S | / |(wx, wy, wz)|
}

// _4
__global__ void llg_quat_apply_4_32(
//	const int nx, const int ny, const int offset,
	const int nxyz,
	float*  wx, float*  wy, float*  wz, float* ww,
 	float* dsx, float* dsy, float* dsz)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i >= nxyz)
		return;

	if(ww[i] == 0)
	{
		return;
	}
	
	dsx[i] = wx[i] * ww[i];
	dsy[i] = wy[i] * ww[i];
	dsz[i] = wz[i] * ww[i];
}

// 	cuda_llg_quat_apply(
// 			    dmdt->d_x,     dmdt->d_y,     dmdt->d_z,     dmdt->d_ms,
//             dmdt->d_hx[T], dmdt->d_hy[T], dmdt->d_hz[T],
// 			dmdt->d_hx[S], dmdt->d_hy[S], dmdt->d_hz[S],
// 			          d_ws1,         d_ws2,         d_ws3,         d_ws4,
// 			alpha, dt, gamma);	

	
void cuda_llg_quat_apply32(const int nx, const int ny, const int nz,
	float* dsx, float* dsy, float* dsz, float* dms, //dest (spinto)
	float* ssx, float* ssy, float* ssz, float* sms, // src (spinfrom)
	float* ddx, float* ddy, float* ddz, float* dds, // dm/dt spins
    float* htx, float* hty, float* htz,              // dm/dt thermal fields
	float* dhx, float* dhy, float* dhz,              // dm/dt fields
	float* ws1, float* ws2, float* ws3, float* ws4,
	const float alpha, const float dt, const float gamma)
{
	// the 0.5 is for the quaternions
	float gadt = (0.5 * gamma * dt) / (1.0 + alpha * alpha);

	const int nxyz = nx*ny*nz;
	const int threads = 512;
	const int blocks = nxyz / threads + 1;

	// _1 calculates rhs of S x (damped field)
	// the (damped field) is done with dm/dt terms
	// result is stored in W
	llg_quat_apply_1_32<<<blocks, threads>>>(nxyz,
					alpha,
					ddx, ddy, ddz, dds,
					dhx, dhy, dhz,
					htx, hty, htz,
					ws1, ws2, ws3, ws4);
	CHECK
	
	// spinfrom x W (via quats)
	llg_quat_apply_2_32<<<blocks, threads>>>(nxyz,
					ssx, ssy, ssz,
					ws1, ws2, ws3, ws4,
					alpha, gadt);
	CHECK

	// normalize
	llg_quat_apply_3_32<<<blocks, threads>>>(nxyz,
					ws1, ws2, ws3, ws4,
					ssx, ssy, ssz, sms,
					dsx, dsy, dsz, dms);
	CHECK
	
	// store in (spinto)
	llg_quat_apply_4_32<<<blocks, threads>>>(nxyz,
					ws1, ws2, ws3, ws4,
					dsx, dsy, dsz);
	CHECK
}

