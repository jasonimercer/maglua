#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
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
	
// 	before hardcode
// 	ptxas info    : Compiling entry function '_Z14llg_quat_applyiiiPdS_S_S_S_S_S_S_S_S_S_ddd' for 'sm_21'
// ptxas info    : Function properties for _Z14llg_quat_applyiiiPdS_S_S_S_S_S_S_S_S_S_ddd
//     80 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
// ptxas info    : Used 44 registers, 8+0 bytes lmem, 160 bytes cmem[0], 144 bytes cmem[2], 28 bytes cmem[16]

// 	after hardcode
// ptxas info    : Compiling entry function '_Z14llg_quat_applyiiiPdS_S_S_S_S_S_S_S_S_S_ddd' for 'sm_21'
// ptxas info    : Function properties for _Z14llg_quat_applyiiiPdS_S_S_S_S_S_S_S_S_S_ddd
//     80 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
// ptxas info    : Used 44 registers, 8+0 bytes lmem, 160 bytes cmem[0], 144 bytes cmem[2], 28 bytes cmem[16]

	
__device__ double4 qconjugate(double4 q)
{
	double4 r;
	r.x = -1.0 * q.x;
	r.y = -1.0 * q.y;
	r.z = -1.0 * q.z;
	r.w =        q.w;
	return r;
}

__device__ double4 qmult(double4 a, double4 b)
{
	double4 ab;

	ab.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
	ab.x = a.w*b.x + b.w*a.x  + a.y*b.z - a.z*b.y;
	ab.y = a.w*b.y + b.w*a.y  + a.z*b.x - a.x*b.z;
	ab.z = a.w*b.z + b.w*a.z  + a.x*b.y - a.y*b.x;

	return ab;
}

// quat mult without calculating the W component
__device__ double4 qmultXYZ(double4 a, double4 b)
{
	double4 ab;

	ab.w = 0;
	ab.x = a.w*b.x + b.w*a.x  + a.y*b.z - a.z*b.y;
	ab.y = a.w*b.y + b.w*a.y  + a.z*b.x - a.x*b.z;
	ab.z = a.w*b.z + b.w*a.z  + a.x*b.y - a.y*b.x;

	return ab;
}

// _1 will compute rhs and store in ws
// dS    -g           a
// -- = ---- S X (H +---S X H)
// dt   1+aa         |S|
__global__ void llg_quat_apply_1(
	const int nx, const int ny, const int nz,
	double alpha,
	double* sx, double* sy, double* sz,
	double* hx, double* hy, double* hz,
	// ws1, ws2, ws3,     ws4
	double* vx, double* vy, double* vz, double* vl) 
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int z = blockDim.z * blockIdx.z + threadIdx.z;
	
	if(x >= nx || y >= ny || z >= nz)
		return;
	
	const int i = x + y*nx + z*nx*ny;
	
	vx[i] = sy[i]*hz[i] - sz[i]*hy[i];
	vy[i] = sz[i]*hx[i] - sx[i]*hz[i];
	vz[i] = sx[i]*hy[i] - sy[i]*hx[i];
	
	vl[i] = sqrt(sx[i]*sx[i] + sy[i]*sy[i] + sz[i]*sz[i]);
	if(vl[i] != 0)
	{
		vl[i] = alpha / vl[i];
	}
	vx[i] = vl[i] * vx[i];
	vy[i] = vl[i] * vy[i];
	vz[i] = vl[i] * vz[i];
	
	vx[i] = hx[i] + vx[i];
	vy[i] = hy[i] + vy[i];
	vz[i] = hz[i] + vz[i];
	vl[i] = sqrt(hx[i]*hx[i] + hy[i]*hy[i] + hz[i]*hz[i]);
}

// _2 will compute the rest
// dS    -g           a
// -- = ---- S X (H +---S X H)
// dt   1+aa         |S|
// the rhs vec and len is in ws
__global__ void llg_quat_apply_2(
	const int nx, const int ny, const int nz,
// 	double* dsx, double* dsy, double* dsz, //dest
	double* ssx, double* ssy, double* ssz, // src
	double* hx, double* hy, double* hz, double* hlen,
	double alpha, double gadt)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int z = blockDim.z * blockIdx.z + threadIdx.z;
	
	if(x >= nx || y >= ny || z >= nz)
		return;

	const int i = x + y*nx + z*nx*ny;

	if(hlen[i] == 0)
	{
// 		dsx[i] = ssx[i];
// 		dsy[i] = ssy[i];
// 		dsz[i] = ssz[i];
		return;
	}

// need to do this later
// 	dms[i] = sms[i];
// 	if(sms[i] == 0)
// 		return;

// dS    -g           a
// -- = ---- S X (H +---S X H)
// dt   1+aa         |S|
	double4 qVec;
	qVec.x = ssx[i]; 
	qVec.y = ssy[i];
	qVec.z = ssz[i];
	qVec.w = 0;
			
	const double theta = hlen[i] * gadt;
	//need to test if sincos works:
	double cost = cos(theta);
	double sint = sin(theta);

	double4 qRot;
	qRot.x = sint * hx[i]; 
	qRot.y = sint * hy[i];
	qRot.z = sint * hz[i];
	qRot.w = cost;

	//this is the rotation: qRes = qRot qVec qRot*
	double4 qRes = qmultXYZ(qmult(qRot, qVec), qconjugate(qRot));

	hx[i] = qRes.x;
	hy[i] = qRes.y;
	hz[i] = qRes.z;
		
		//need to normalize later
// 		const double il = FAST_DIV(1.0, sqrt(dsx[i]*dsx[i] + dsy[i]*dsy[i] + dsz[i]*dsz[i]));
		
// 		dsx[i] = dsx[i] * il * sms[i];
// 		dsy[i] = dsy[i] * il * sms[i];
// 		dsz[i] = dsz[i] * il * sms[i]; 
}

// _3 normalize
__global__ void llg_quat_apply_3(
	const int nx, const int ny, const int nz,
	double*  hx, double*  hy, double*  hz, double* hlen,
	double* ssx, double* ssy, double* ssz, double* sms,
 	double* dsx, double* dsy, double* dsz, double* dms)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int z = blockDim.z * blockIdx.z + threadIdx.z;
	
	if(x >= nx || y >= ny || z >= nz)
		return;

	const int i = x + y*nx + z*nx*ny;

	dms[i] = sms[i];
	
	if(hlen[i] == 0)
	{
		dsx[i] = ssx[i];
		dsy[i] = ssy[i];
		dsz[i] = ssz[i];
		return;
	}
	
	//using hlen as temp var. save a reg?
	hlen[i] = sqrt(hx[i]*hx[i] + hy[i]*hy[i] + hz[i]*hz[i]);
	
	hlen[i] = sms[i] / hlen[i];
}

// _4
__global__ void llg_quat_apply_4(
	const int nx, const int ny, const int nz,
	double*  hx, double*  hy, double*  hz, double* hlen,
 	double* dsx, double* dsy, double* dsz)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	const int y = blockDim.y * blockIdx.y + threadIdx.y;
	const int z = blockDim.z * blockIdx.z + threadIdx.z;
	
	if(x >= nx || y >= ny || z >= nz)
		return;

	const int i = x + y*nx + z*nx*ny;

	dsx[i] = hx[i] * hlen[i];
	dsy[i] = hy[i] * hlen[i];
	dsz[i] = hz[i] * hlen[i];
}



void cuda_llg_quat_apply(const int nx, const int ny, const int nz,
	double* dsx, double* dsy, double* dsz, double* dms, //dest
	double* ssx, double* ssy, double* ssz, double* sms, // src
	double* hx, double* hy, double* hz,
	double* ws1, double* ws2, double* ws3, double* ws4,
	const double alpha, const double dt, const double gamma)
{
	double gadt = (gamma * dt) / (2.0 - 2.0 * alpha * alpha);
	if(nz == 1)
	{
		const int blocksx = nx / 32 + 1;
		const int blocksy = ny / 32 + 1;
	
		dim3 blocks(blocksx, blocksy);
		dim3 threads(32, 32);
		
		llg_quat_apply_1<<<blocks, threads>>>(
						nx, ny, nz,
						alpha,
						ssx, ssy, ssz, 
						 hx,  hy,  hz,
						ws1, ws2, ws3, ws4);
		CHECK
		
		llg_quat_apply_2<<<blocks, threads>>>(nx, ny, nz,
						ssx, ssy, ssz,
						ws1, ws2, ws3, ws4,
						alpha, gadt);
		CHECK

		llg_quat_apply_3<<<blocks, threads>>>(nx, ny, nz,
						ws1, ws2, ws3, ws4,
						ssx, ssy, ssz, sms,
						dsx, dsy, dsz, dms);
		CHECK
		
		llg_quat_apply_4<<<blocks, threads>>>(nx, ny, nz,
						ws1, ws2, ws3, ws4,
						dsx, dsy, dsz);
		CHECK
	}
	else
	{
#warning need to implement 3D
	}}

