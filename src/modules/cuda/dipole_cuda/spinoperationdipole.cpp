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

#include "spinoperationdipole.h"
#include "spinsystem.h"
#include "dipolesupport.h"
#include "info.h"

#include <stdlib.h>
#include <math.h>

DipoleCuda::DipoleCuda(int nx, int ny, int nz)
	: SpinOperation("DipolePureCuda", DIPOLE_SLOT, nx, ny, nz, ENCODE_DIPOLE)
{
	g = 1;
	gmax = 2000;

	ABC[0] = 1; ABC[1] = 0; ABC[2] = 0;
	ABC[3] = 0; ABC[4] = 1; ABC[5] = 0;
	ABC[6] = 0; ABC[7] = 0; ABC[8] = 1;

	plan = 0;
}

void DipoleCuda::init()
{
	getPlan();
}

void DipoleCuda::deinit()
{
	if(plan)
	{
		free_JM_LONGRANGE_PLAN(plan);
		plan = 0;
	}	
}

void DipoleCuda::encode(buffer* b)
{
	encodeInteger(nx, b);
	encodeInteger(ny, b);
	encodeInteger(nz, b);
	encodeInteger(gmax, b);

	for(int i=0; i<9; i++)
	{
		encodeDouble(ABC[i], b);
	}
}

int  DipoleCuda::decode(buffer* b)
{
	deinit();

	nx = decodeInteger(b);
	ny = decodeInteger(b);
	nz = decodeInteger(b);
	gmax = decodeInteger(b);
	nxyz = nx*ny*nz;

	for(int i=0; i<9; i++)
	{
		ABC[i] = decodeDouble(b);
	}


	return 0;
}

DipoleCuda::~DipoleCuda()
{
	deinit();
}

void DipoleCuda::getPlan()
{
	deinit();
	
	int s = nx*ny * (nz*2-1);
	double* XX = new double[s];
	double* XY = new double[s];
	double* XZ = new double[s];
	double* YY = new double[s];
	double* YZ = new double[s];
	double* ZZ = new double[s];
	
	dipoleLoad(
		nx, ny, nz,
		gmax, ABC,
		XX, XY, XZ,
		YY, YZ, ZZ);
	
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
}

	
bool DipoleCuda::apply(SpinSystem* ss)
{
	markSlotUsed(ss);

	if(!plan)
		getPlan();
	
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

// 	ss->new_host_fields[slot] = false;
	ss->new_device_fields[slot] = true;

// 	ss->sync_fields_dh(slot);
// 	printf("(%s:%i) %f %f %f\n", __FILE__, __LINE__, ss->h_hx[slot][1], ss->h_hy[slot][1], ss->h_hz[slot][1]);
	
	return true;
}







DipoleCuda* checkDipoleCuda(lua_State* L, int idx)
{
	DipoleCuda** pp = (DipoleCuda**)luaL_checkudata(L, idx, "MERCER.dipole");
    luaL_argcheck(L, pp != NULL, 1, "`DipoleCuda' expected");
    return *pp;
}

void lua_pushDipoleCuda(lua_State* L, DipoleCuda* dip)
{
	dip->refcount++;
	DipoleCuda** pp = (DipoleCuda**)lua_newuserdata(L, sizeof(DipoleCuda**));
	
	*pp = dip;
	luaL_getmetatable(L, "MERCER.dipole");
	lua_setmetatable(L, -2);
}

int l_dip_new(lua_State* L)
{
	int n[3];
	lua_getnewargs(L, n, 1);

	lua_pushDipoleCuda(L, new DipoleCuda(n[0], n[1], n[2]));
	return 1;
}


int l_dip_setstrength(lua_State* L)
{
	DipoleCuda* dip = checkDipoleCuda(L, 1);
	if(!dip) return 0;

	dip->g = lua_tonumber(L, 2);
	return 0;
}

int l_dip_gc(lua_State* L)
{
	DipoleCuda* dip = checkDipoleCuda(L, 1);
	if(!dip) return 0;

	dip->refcount--;
	if(dip->refcount == 0)
		delete dip;
	
	return 0;
}

int l_dip_apply(lua_State* L)
{
// 	printf("(%s:%i)\n", __FILE__, __LINE__);
	DipoleCuda* dip = checkDipoleCuda(L, 1);
// 	printf("(%s:%i)\n", __FILE__, __LINE__);
	if(!dip) return 0;
// 	printf("(%s:%i)\n", __FILE__, __LINE__);
	SpinSystem* ss = checkSpinSystem(L, 2);
// 	printf("(%s:%i)\n", __FILE__, __LINE__);
	
	if(!dip->apply(ss))
	{
		printf("(%s:%i)\n", __FILE__, __LINE__);
		return luaL_error(L, dip->errormsg.c_str());
	}

	return 0;
}

int l_dip_getstrength(lua_State* L)
{
	DipoleCuda* dip = checkDipoleCuda(L, 1);
	if(!dip) return 0;

	lua_pushnumber(L, dip->g);

	return 1;
}
int l_dip_setunitcell(lua_State* L)
{
	DipoleCuda* dip = checkDipoleCuda(L, 1);
	if(!dip) return 0;

	double A[3];
	double B[3];
	double C[3];
	
	int r1 = lua_getNdouble(L, 3, A, 2, 0);
	int r2 = lua_getNdouble(L, 3, B, 2+r1, 0);
	int r3 = lua_getNdouble(L, 3, C, 2+r2+r3, 0);
	
	for(int i=0; i<3; i++)
	{
		dip->ABC[i+0] = A[i];
		dip->ABC[i+3] = B[i];
		dip->ABC[i+6] = C[i];
	}

	return 0;
}
int l_dip_getunitcell(lua_State* L)
{
	DipoleCuda* dip = checkDipoleCuda(L, 1);
	if(!dip) return 0;

	double* ABC[3];
	ABC[0] = &(dip->ABC[0]);
	ABC[1] = &(dip->ABC[3]);
	ABC[2] = &(dip->ABC[6]);
	
	for(int i=0; i<3; i++)
	{
		lua_newtable(L);
		for(int j=0; j<3; j++)
		{
			lua_pushinteger(L, j+1);
			lua_pushnumber(L, ABC[i][j]);
			lua_settable(L, -3);
		}
	}
	
	return 3;
}
int l_dip_settrunc(lua_State* L)
{
	DipoleCuda* dip = checkDipoleCuda(L, 1);
	if(!dip) return 0;

	dip->gmax = lua_tointeger(L, 2);

	return 0;
}
int l_dip_gettrunc(lua_State* L)
{
	DipoleCuda* dip = checkDipoleCuda(L, 1);
	if(!dip) return 0;

	lua_pushnumber(L, dip->gmax);

	return 1;
}

static int l_dip_tostring(lua_State* L)
{
	DipoleCuda* dip = checkDipoleCuda(L, 1);
	if(!dip) return 0;
	
	lua_pushfstring(L, "DipoleCuda (%dx%dx%d)", dip->nx, dip->ny, dip->nz);
	
	return 1;
}

static int l_dip_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.dipole");
	return 1;
}

static int l_dip_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates the dipolar field of a *SpinSystem*");
		lua_pushstring(L, ""); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	if(lua_istable(L, 1))
	{
		return 0;
	}
	
	if(!lua_iscfunction(L, 1))
	{
		return luaL_error(L, "help expect zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if(func == l_dip_new)
	{
		lua_pushstring(L, "Create a new DipoleCuda Operator.");
		lua_pushstring(L, "1 *3Vector*: system size"); 
		lua_pushstring(L, "1 DipoleCuda object");
		return 3;
	}
	
	
	if(func == l_dip_apply)
	{
		lua_pushstring(L, "Calculate the dipolar field of a *SpinSystem*");
		lua_pushstring(L, "1 *SpinSystem*: This spin system will receive the field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_dip_setstrength)
	{
		lua_pushstring(L, "Set the strength of the Dipolar Field");
		lua_pushstring(L, "1 number: strength of the field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_dip_getstrength)
	{
		lua_pushstring(L, "Get the strength of the Dipolar Field");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 number: strength of the field");
		return 3;
	}
	
	if(func == l_dip_setunitcell)
	{
		lua_pushstring(L, "Set the unit cell of a lattice site");
		lua_pushstring(L, "3 *3Vector*: The A, B and C vectors defining the unit cell. By default, this is {1,0,0},{0,1,0},{0,0,1} or a cubic system.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_dip_getunitcell)
	{
		lua_pushstring(L, "Get the unit cell of a lattice site");
		lua_pushstring(L, "");
		lua_pushstring(L, "3 tables: The A, B and C vectors defining the unit cell. By default, this is {1,0,0},{0,1,0},{0,0,1} or a cubic system.");
		return 3;
	}

	if(func == l_dip_settrunc)
	{
		lua_pushstring(L, "Set the truncation distance in spins of the dipolar sum.");
		lua_pushstring(L, "1 Integers: Radius of spins to sum out to.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_dip_gettrunc)
	{
		lua_pushstring(L, "Get the truncation distance in spins of the dipolar sum.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integers: Radius of spins to sum out to.");
		return 3;
	}

	return 0;
}


void registerDipoleCuda(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_dip_gc},
		{"__tostring",   l_dip_tostring},
		{"apply",        l_dip_apply},
		{"setStrength",  l_dip_setstrength},
		{"strength",     l_dip_getstrength},
		{"setUnitCell",  l_dip_setunitcell},
		{"unitCell",     l_dip_getunitcell},
		{"setTruncation",l_dip_settrunc},
		{"truncation",   l_dip_gettrunc},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.dipole");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_dip_new},
		{"help",                l_dip_help},
		{"metatable",           l_dip_mt},
		{NULL, NULL}
	};
		
	luaL_register(L, "Dipole", functions);
	lua_pop(L,1);	
}



extern "C"
{
DIPOLECUDA_API int lib_register(lua_State* L);
DIPOLECUDA_API int lib_version(lua_State* L);
DIPOLECUDA_API const char* lib_name(lua_State* L);
DIPOLECUDA_API void lib_main(lua_State* L, int argc, char** argv);
}

DIPOLECUDA_API int lib_register(lua_State* L)
{
	registerDipoleCuda(L);
	return 0;
}


DIPOLECUDA_API int lib_version(lua_State* L)
{
	return __revi;
}


DIPOLECUDA_API const char* lib_name(lua_State* L)
{
	return "Dipole-Cuda";
}

DIPOLECUDA_API void lib_main(lua_State* L, int argc, char** argv)
{
}


