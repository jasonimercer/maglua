/******************************************************************************
* Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#include "spinoperationlongrange.h"
#include "spinsystem.h"
#include "spinsystem.hpp"
#include "info.h"

#include <stdlib.h>
#include <math.h>

LongRangeCuda::LongRangeCuda(const char* name, const int field_slot, int nx, int ny, int nz, const int encode_tag)
	: SpinOperation(name, field_slot, nx, ny, nz, encode_tag)
{
	g = 1;
	gmax = 2000;

	ABC[0] = 1; ABC[1] = 0; ABC[2] = 0;
	ABC[3] = 0; ABC[4] = 1; ABC[5] = 0;
	ABC[6] = 0; ABC[7] = 0; ABC[8] = 1;

	matrixLoaded = false;
	plan = 0;
	XX = 0;
	registerWS();
}

void LongRangeCuda::init()
{
    if(XX) return;
	deinit();
	int s = nx*ny * (nz);// *2-1);
	XX = new double[s];
	XY = new double[s];
	XZ = new double[s];
	YY = new double[s];
	YZ = new double[s];
	ZZ = new double[s];

// 	getPlan();
}

static int offsetOK(int nx, int ny, int nz,  int x, int y, int z, int& offset)
{
	if(x<0 || x >= nx) return 0;
	if(y<0 || y >= ny) return 0;
	if(z<0 || z >= nz) return 0;
	
	offset = x + y*nx + z*nx*ny;
	return 1;
}

#define getsetPattern(AB) \
double LongRangeCuda::get##AB (int ox, int oy, int oz) \
{ \
    ox = (ox + 10*nx)%nx; \
    oy = (oy + 10*ny)%ny; \
    loadMatrix(); \
	int offset; \
	if(offsetOK(nx,ny,nz, ox,oy,oz, offset)) \
		return AB [offset]; \
	return 0; \
} \
 \
void   LongRangeCuda::set##AB (int ox, int oy, int oz, double value) \
{ \
    ox = (ox + 10*nx)%nx; \
    oy = (oy + 10*ny)%ny; \
	loadMatrix(); \
	int offset; \
	if(offsetOK(nx,ny,nz, ox,oy,oz, offset)) \
	{ \
		AB [offset] = value; \
		newHostData = true; \
	} \
} 

getsetPattern(XX)
getsetPattern(XY)
getsetPattern(XZ)
getsetPattern(YY)
getsetPattern(YZ)
getsetPattern(ZZ)

double LongRangeCuda::getAB(int matrix, int ox, int oy, int oz)
{
	switch(matrix)
	{
		case 0:	return getXX(ox,oy,oz);
		case 1:	return getXY(ox,oy,oz);
		case 2:	return getXZ(ox,oy,oz);
		case 3:	return getYY(ox,oy,oz);
		case 4:	return getYZ(ox,oy,oz);
		case 5:	return getZZ(ox,oy,oz);
	}
	return 0;
}

void  LongRangeCuda::setAB(int matrix, int ox, int oy, int oz, double value)
{
	switch(matrix)
	{
		case 0:	setXX(ox,oy,oz,value); break;
		case 1:	setXY(ox,oy,oz,value); break;
		case 2:	setXZ(ox,oy,oz,value); break;
		case 3:	setYY(ox,oy,oz,value); break;
		case 4:	setYZ(ox,oy,oz,value); break;
		case 5:	setZZ(ox,oy,oz,value); break;
	}
}
void LongRangeCuda::deinit()
{
	if(plan)
	{
		free_JM_LONGRANGE_PLAN(plan);
		plan = 0;
	}	
	if(XX)
	{
		delete [] XX;
		delete [] XY;
		delete [] XZ;
		delete [] YY;
		delete [] YZ;
		delete [] ZZ;
		XX = 0;
	}
}

LongRangeCuda::~LongRangeCuda()
{
	unregisterWS();
	deinit();
}

void LongRangeCuda::loadMatrix()
{
	if(matrixLoaded) return;
	init();
	loadMatrixFunction(XX, XY, XZ, YY, YZ, ZZ); //implemented by child

	matrixLoaded = true;
}

	
bool LongRangeCuda::getPlan()
{
// 	deinit();
/*	int s = nx*ny * (nz);// *2-1);
	double* XX = new double[s];
	double* XY = new double[s];
	double* XZ = new double[s];
	double* YY = new double[s];
	double* YZ = new double[s];
	double* ZZ = new double[s];*/

	loadMatrix(); //only fires once

// 	loadMatrixFunction(XX, XY, XZ, YY, YZ, ZZ); //implemented by child
	
	if(plan)
	{
		free_JM_LONGRANGE_PLAN(plan);
		plan = 0;
	}	
	
	const int sz = JM_LONGRANGE_PLAN_ws_size(nx, ny, nz);
	
	void* ws_d_A;
	void* ws_d_B;
	
	getWSMem(&ws_d_A, sz, &ws_d_B, sz);
	

	
	plan = make_JM_LONGRANGE_PLAN(nx, ny, nz,
								  XX, XY, XZ,
									  YY, YZ,
									      ZZ, ws_d_A, ws_d_B);

// 	delete [] XX;
// 	delete [] XY;
// 	delete [] XZ;
// 	delete [] YY;
// 	delete [] YZ;
// 	delete [] ZZ;
	
	if(!plan)
	{
		errormsg = "Failed to factor system into small primes\n";
	}
	
	return plan != 0;
}

bool LongRangeCuda::applyToSum(SpinSystem* ss)
{
	if(newHostData)
		getPlan();
	if(!plan)
		getPlan();
	if(!plan)
		return false;
	
	ss->sync_spins_hd();

	int conv_ws = JM_LONGRANGE_PLAN_ws_size(nx, ny, nz);
	int field_ws = sizeof(double)*nx*ny*nz;
	
	void* d_ws_A;
	void* d_ws_B;
	
	double* d_wsx;
	double* d_wsy;
	double* d_wsz;
	
	getWSMem((void**)&d_ws_A, conv_ws, 
			 (void**)&d_ws_B, conv_ws, 
			 (void**)&d_wsx,  field_ws,
			 (void**)&d_wsy,  field_ws,
			 (void**)&d_wsz,  field_ws);
		  
// 	void* ws = getWSMem(conv_ws + field_ws*3);
	
// 	char* cws = (char*)ws;
// 	double* d_wsx = (double*)(& cws[conv_ws + field_ws * 0]); //dumb looking
// 	double* d_wsy = (double*)(& cws[conv_ws + field_ws * 1]); //but gets rid of pointer
// 	double* d_wsz = (double*)(& cws[conv_ws + field_ws * 2]); //math warning
	
	const double* d_sx = ss->d_x;
	const double* d_sy = ss->d_y;
	const double* d_sz = ss->d_z;
	
	JM_LONGRANGE(plan, 
					d_sx, d_sy, d_sz, 
					d_wsx, d_wsy, d_wsz, d_ws_A, d_ws_B);

	cuda_scaledAddArrays(ss->d_hx[SUM_SLOT], nx*ny*nz, 1.0, ss->d_hx[SUM_SLOT], g, d_wsx);
	cuda_scaledAddArrays(ss->d_hy[SUM_SLOT], nx*ny*nz, 1.0, ss->d_hy[SUM_SLOT], g, d_wsy);
	cuda_scaledAddArrays(ss->d_hz[SUM_SLOT], nx*ny*nz, 1.0, ss->d_hz[SUM_SLOT], g, d_wsz);
	ss->slot_used[SUM_SLOT] = true;
	
	return true;
}

bool LongRangeCuda::apply(SpinSystem* ss)
{
	markSlotUsed(ss);

	if(!plan)
		getPlan();
	if(!plan)
		return false;
	
	ss->sync_spins_hd();
	
	double* d_hx = ss->d_hx[slot];
	double* d_hy = ss->d_hy[slot];
	double* d_hz = ss->d_hz[slot];
	
	const double* d_sx = ss->d_x;
	const double* d_sy = ss->d_y;
	const double* d_sz = ss->d_z;

	const int sz = JM_LONGRANGE_PLAN_ws_size(nx, ny, nz);
// 	void* ws = getWSMem(sz);
	
	void* ws_d_A;
	void* ws_d_B;
	
	getWSMem(&ws_d_A, sz, &ws_d_B, sz);
	
	
	JM_LONGRANGE(plan, 
					d_sx, d_sy, d_sz, 
					d_hx, d_hy, d_hz, ws_d_A, ws_d_B);

	ss_d_scale3DArray(d_hx, nxyz, g);
	ss_d_scale3DArray(d_hy, nxyz, g);
	ss_d_scale3DArray(d_hz, nxyz, g);
	
	ss->new_device_fields[slot] = true;

	//	ss->sync_spins_dh();
	// 	printf("(%s:%i) %f %f %f\n", __FILE__, __LINE__, ss->h_x[1], ss->h_y[1], ss->h_z[1]);

//  	ss->sync_fields_dh(slot);
// 	const int i = 16*16;
// 	printf("(%s:%i) %f %f %f\n", __FILE__, __LINE__, ss->h_hx[slot][i], ss->h_hy[slot][i], ss->h_hz[slot][i]);
	
	return true;
}






extern "C"
{
LONGRANGECUDA_API int lib_register(lua_State* L);
LONGRANGECUDA_API int lib_version(lua_State* L);
LONGRANGECUDA_API const char* lib_name(lua_State* L);
LONGRANGECUDA_API int lib_main(lua_State* L);
}

LONGRANGECUDA_API int lib_register(lua_State* L)
{
	return 0;
}

LONGRANGECUDA_API int lib_version(lua_State* L)
{
	return __revi;
}

LONGRANGECUDA_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "LongRange-Cuda";
#else
	return "LongRange-Cuda-Debug";
#endif
}

LONGRANGECUDA_API int lib_main(lua_State* L)
{
	return 0;
}
