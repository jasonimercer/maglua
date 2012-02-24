/******************************************************************************
* Copyright (C) 2008-2010 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/
// spinoperationmagnetostatics
#include "spinoperationmagnetostatics.h"
#include "spinsystem.h"
#include "magnetostaticssupport.h"
#include "info.h"

#include <stdlib.h>
#include <math.h>

Magnetostatic::Magnetostatic(int nx, int ny, int nz)
	: LongRange("Magnetostatic", DIPOLE_SLOT, nx, ny, nz, ENCODE_MAGNETOSTATIC)
{
	crossover_tolerance = 0.0001;

	volumeDimensions[0] = 1;	
	volumeDimensions[1] = 1;	
	volumeDimensions[2] = 1;	
}

void Magnetostatic::encode(buffer* b)
{
	encodeInteger(nx, b);
	encodeInteger(ny, b);
	encodeInteger(nz, b);
	encodeInteger(gmax, b);
	encodeDouble(g, b);
	
	for(int i=0; i<3; i++)
		encodeDouble(volumeDimensions[i], b);
	
	for(int i=0; i<9; i++)
		encodeDouble(ABC[i], b);
	
	encodeDouble(crossover_tolerance, b);	
}

int  Magnetostatic::decode(buffer* b)
{
	deinit();

	nx = decodeInteger(b);
	ny = decodeInteger(b);
	nz = decodeInteger(b);
	gmax = decodeInteger(b);
	nxyz = nx*ny*nz;
	g = decodeDouble(b);
	
	for(int i=0; i<3; i++)
		volumeDimensions[i] = decodeDouble(b);
	
	for(int i=0; i<9; i++)
	{
		ABC[i] = decodeDouble(b);
	}
	
	crossover_tolerance = decodeDouble(b);
	return 0;
}





Magnetostatic::~Magnetostatic()
{
	deinit();
}

void Magnetostatic::loadMatrixFunction(double* XX, double* XY, double* XZ, double* YY, double* YZ, double* ZZ)
{
	magnetostaticsLoad(
		nx, ny, nz,
		gmax, ABC,
		volumeDimensions,
		XX, XY, XZ,
		YY, YZ, ZZ, crossover_tolerance);
}







Magnetostatic* checkMagnetostatic(lua_State* L, int idx)
{
	Magnetostatic** pp = (Magnetostatic**)luaL_checkudata(L, idx, "MERCER.magnetostatics");
    luaL_argcheck(L, pp != NULL, 1, "`Magnetostatic' expected");
    return *pp;
}

void lua_pushMagnetostatic(lua_State* L, Encodable* _mag)
{
	Magnetostatic* mag = dynamic_cast<Magnetostatic*>(_mag);
	if(!mag) return;
	mag->refcount++;
	Magnetostatic** pp = (Magnetostatic**)lua_newuserdata(L, sizeof(Magnetostatic**));
	
	*pp = mag;
	luaL_getmetatable(L, "MERCER.magnetostatics");
	lua_setmetatable(L, -2);
}

int l_mag_new(lua_State* L)
{
	int n[3];
	lua_getnewargs(L, n, 1);

	lua_pushMagnetostatic(L, new Magnetostatic(n[0], n[1], n[2]));
	return 1;
}


int l_mag_setstrength(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	mag->g = lua_tonumber(L, 2);
	return 0;
}

int l_mag_gc(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	mag->refcount--;
	if(mag->refcount == 0)
		delete mag;
	
	return 0;
}

int l_mag_apply(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;
	SpinSystem* ss = checkSpinSystem(L, 2);
	
	if(!mag->apply(ss))
		return luaL_error(L, mag->errormsg.c_str());
	
	return 0;
}

int l_mag_getstrength(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	lua_pushnumber(L, mag->g);

	return 1;
}
int l_mag_setunitcell(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	double A[3];
	double B[3];
	double C[3];
	
	int r1 = lua_getNdouble(L, 3, A, 2, 0);
	int r2 = lua_getNdouble(L, 3, B, 2+r1, 0);
	int r3 = lua_getNdouble(L, 3, C, 2+r1+r2, 0);
	
	for(int i=0; i<3; i++)
	{
		mag->ABC[i+0] = A[i];
		mag->ABC[i+3] = B[i];
		mag->ABC[i+6] = C[i];
	}

	return 0;
}

int l_mag_getunitcell(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	for(int i=0; i<9; i++)
		lua_pushnumber(L, mag->ABC[i]);

	return 9;
}
static int l_mag_settrunc(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	lua_getglobal(L, "math");
	lua_pushstring(L, "huge");
	lua_gettable(L, -2);
	lua_pushvalue(L, 2);
	int huge = lua_equal(L, -2, -1);
	
	if(huge)
	{
		mag->gmax = -1;
	}
	else
	{
		mag->gmax = lua_tointeger(L, 2);
	}
	return 0;
}
static int l_mag_gettrunc(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	if(mag->gmax == -1)
	{
		lua_getglobal(L, "math");
		lua_pushstring(L, "huge");
		lua_gettable(L, -2);
		lua_remove(L, -2);//remove table (not really needed);
	}
	else
		lua_pushnumber(L, mag->gmax);

	return 1;
}

static int l_mag_tostring(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;
	
	lua_pushfstring(L, "Magnetostatic (%dx%dx%d)", mag->nx, mag->ny, mag->nz);
	
	return 1;
}

static int l_mag_setcelldims(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;

	if(lua_getNdouble(L, 3, mag->volumeDimensions, 2, 1) < 0)
		return luaL_error(L, "Magnetostatic.setCellDimensions requires 3 values");

	return 0;
}

static int l_mag_getcelldims(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;
	
	for(int i=0; i<3; i++)
		lua_pushnumber(L, mag->volumeDimensions[i]);

	return 3;
}

static int l_mag_setcrossover(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;
	
	mag->crossover_tolerance = lua_tonumber(L, 2);
	return 0;
}

static int l_mag_getcrossover(lua_State* L)
{
	Magnetostatic* mag = checkMagnetostatic(L, 1);
	if(!mag) return 0;
	
	lua_pushnumber(L, mag->crossover_tolerance);
	return 1;
}



static int l_mag_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.magnetostatics");
	return 1;
}

static int l_mag_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates the Magnetostatic field of a *SpinSystem*");
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
	
	if(func == l_mag_new)
	{
		lua_pushstring(L, "Create a new Magnetostatic Operator.");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 Magnetostatic object");
		return 3;
	}
	
	
	if(func == l_mag_apply)
	{
		lua_pushstring(L, "Calculate the Magnetostatic field of a *SpinSystem*");
		lua_pushstring(L, "1 *SpinSystem*: This spin system will receive the field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_mag_setstrength)
	{
		lua_pushstring(L, "Set the strength of the Dipolar Field");
		lua_pushstring(L, "1 number: strength of the field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_mag_getstrength)
	{
		lua_pushstring(L, "Get the strength of the Dipolar Field");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 number: strength of the field");
		return 3;
	}
	
	if(func == l_mag_setunitcell)
	{
		lua_pushstring(L, "Set the unit cell of a lattice site");
		lua_pushstring(L, "9 numbers: The A, B and C vectors defining the unit cell. By default, this is (1,0,0,0,1,0,0,0,1) or a cubic system.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_mag_getunitcell)
	{
		lua_pushstring(L, "Get the unit cell of a lattice site");
		lua_pushstring(L, "");
		lua_pushstring(L, "9 numbers: The A, B and C vectors defining the unit cell. By default, this is (1,0,0,0,1,0,0,0,1) or a cubic system.");
		return 3;
	}

	if(func == l_mag_settrunc)
	{
		lua_pushstring(L, "Set the truncation distance in spins of the Magnetostatic sum.");
		lua_pushstring(L, "1 Integers: Radius of spins to sum out to.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_mag_gettrunc)
	{
		lua_pushstring(L, "Get the truncation distance in spins of the Magnetostatic sum.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integers: Radius of spins to sum out to.");
		return 3;
	}
	
	if(func == l_mag_setcelldims)
	{
		lua_pushstring(L, "Set the dimension of each Rectangular Prism");
		lua_pushstring(L, "1 *3Vector*: The x, y and z lengths of the prism");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_mag_getcelldims)
	{
		lua_pushstring(L, "Get the dimension of each Rectangular Prism");
		lua_pushstring(L, "");
		lua_pushstring(L, "3 Numbers: The x, y and z lengths of the prism");
		return 3;
	}
	
	if(func == l_mag_setcrossover)
	{
		lua_pushstring(L, "Set the relative error to define the crossover from magnetostatics to dipole calculations in the interaction matrix generation. Initial value is 0.0001.");
		lua_pushstring(L, "1 Number: The relative error for the crossover");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_mag_getcrossover)
	{
		lua_pushstring(L, "Get the relative error to define the crossover from magnetostatics to dipole calculations in the interaction matrix generation. Initial value is 0.0001.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: The relative error for the crossover");
		return 3;
	}
	
	return 0;
}

static Encodable* newThing()
{
	return new Magnetostatic;
}


void registerMagnetostatic(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_mag_gc},
		{"__tostring",   l_mag_tostring},
		{"apply",        l_mag_apply},
		{"setStrength",  l_mag_setstrength},
		{"strength",     l_mag_getstrength},

		{"setCellDimensions", l_mag_setcelldims},
		{"cellDimensions",    l_mag_getcelldims},

		{"setUnitCell",  l_mag_setunitcell},
		{"unitCell",     l_mag_getunitcell},
		
		{"setTruncation",l_mag_settrunc},
		{"truncation",   l_mag_gettrunc},
		
		{"setCrossoverTolerance", l_mag_setcrossover},
		{"crossoverTolerance", l_mag_setcrossover},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.magnetostatics");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_mag_new},
		{"help",                l_mag_help},
		{"metatable",           l_mag_mt},
		{NULL, NULL}
	};
		
	luaL_register(L, "Magnetostatic", functions);
	lua_pop(L,1);	

	Factory_registerItem(ENCODE_MAGNETOSTATIC, newThing, lua_pushMagnetostatic, "Magnetostatic");

}


#ifdef WIN32
 #ifdef MAGNETOSTATICS_EXPORTS
  #define MAGNETOSTATICS_API __declspec(dllexport)
 #else
  #define MAGNETOSTATICS_API __declspec(dllimport)
 #endif
#else
 #define MAGNETOSTATICS_API 
#endif


extern "C"
{
MAGNETOSTATICS_API int lib_register(lua_State* L);
MAGNETOSTATICS_API int lib_version(lua_State* L);
MAGNETOSTATICS_API const char* lib_name(lua_State* L);
MAGNETOSTATICS_API int lib_main(lua_State* L);
	
}

MAGNETOSTATICS_API int lib_register(lua_State* L)
{
	registerMagnetostatic(L);
	return 0;
}

MAGNETOSTATICS_API int lib_version(lua_State* L)
{
	return __revi;
}

MAGNETOSTATICS_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Magnetostatics";
#else
	return "Magnetostatics-Debug";
#endif
}

MAGNETOSTATICS_API int lib_main(lua_State* L)
{
	return 0;
}




