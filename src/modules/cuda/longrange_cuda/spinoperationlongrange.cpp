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

	plan = 0;
}

void LongRangeCuda::init()
{
	getPlan();
}

void LongRangeCuda::deinit()
{
	if(plan)
	{
		free_JM_LONGRANGE_PLAN(plan);
		plan = 0;
	}	
}

LongRangeCuda::~LongRangeCuda()
{
	deinit();
}

bool LongRangeCuda::getPlan()
{
	deinit();
	
	int s = nx*ny * (nz);// *2-1);
	double* XX = new double[s];
	double* XY = new double[s];
	double* XZ = new double[s];
	double* YY = new double[s];
	double* YZ = new double[s];
	double* ZZ = new double[s];
	
	loadMatrixFunction(XX, XY, XZ, YY, YZ, ZZ); //implemented by child
	
	plan = make_JM_LONGRANGE_PLAN(nx, ny, nz,
								  XX, XY, XZ,
									  YY, YZ,
									      ZZ);

	delete [] XX;
	delete [] XY;
	delete [] XZ;
	delete [] YY;
	delete [] YZ;
	delete [] ZZ;
	
	if(!plan)
	{
		errormsg = "Failed to factor system into small primes\n";
	}
	
	return plan != 0;
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
	
	JM_LONGRANGE(plan, 
					d_sx, d_sy, d_sz, 
					d_hx, d_hy, d_hz);

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
LONGRANGECUDA_API int lib_main(lua_State* L, int argc, char** argv);
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
	return "LongRange-Cuda";
}

LONGRANGECUDA_API int lib_main(lua_State* L, int argc, char** argv)
{
	return 0;
}
