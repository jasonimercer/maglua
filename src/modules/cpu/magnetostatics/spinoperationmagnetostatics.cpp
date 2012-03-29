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
#ifndef WIN32
#include <strings.h>
#endif

#include <stdlib.h>
#include <math.h>

#if defined NDEBUG || defined __OPTIMIZE__
#define DDD
#else
#define DDD printf("(%s:%i)\n", __FILE__, __LINE__);
#endif



Magnetostatic::Magnetostatic(int nx, int ny, int nz)
	: LongRange(Magnetostatic::typeName(), DIPOLE_SLOT, nx, ny, nz, hash32(Magnetostatic::typeName()))
{
	crossover_tolerance = 0.0001;

	volumeDimensions[0] = 1;	
	volumeDimensions[1] = 1;	
	volumeDimensions[2] = 1;	
}

int Magnetostatic::luaInit(lua_State* L)
{
	return LongRange::luaInit(L); //gets nx, ny, nz, nxyz
}

void Magnetostatic::push(lua_State* L)
{
	luaT_push<Magnetostatic>(L, this);
}

void Magnetostatic::encode(buffer* b)
{
	encodeInteger(nx, b);
	encodeInteger(ny, b);
	encodeInteger(nz, b);
	encodeInteger(gmax, b);
	encodeDouble(g, b);
	encodeDouble(global_scale, b);

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
	global_scale = decodeDouble(b);

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






static int l_setcelldims(lua_State* L)
{
	LUA_PREAMBLE(Magnetostatic, mag, 1);

	if(lua_getNdouble(L, 3, mag->volumeDimensions, 2, 1) < 0)
		return luaL_error(L, "Magnetostatic.setCellDimensions requires 3 values");

	return 0;
}

static int l_getcelldims(lua_State* L)
{
	LUA_PREAMBLE(Magnetostatic, mag, 1);
	
	for(int i=0; i<3; i++)
		lua_pushnumber(L, mag->volumeDimensions[i]);

	return 3;
}

static int l_setcrossover(lua_State* L)
{
	LUA_PREAMBLE(Magnetostatic, mag, 1);
	mag->crossover_tolerance = lua_tonumber(L, 2);
	return 0;
}

static int l_getcrossover(lua_State* L)
{
	LUA_PREAMBLE(Magnetostatic, mag, 1);
	lua_pushnumber(L, mag->crossover_tolerance);
	return 1;
}


int Magnetostatic::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates the magnetostatic field of a *SpinSystem*");
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}

	if(lua_istable(L, 1))
	{
		return 0;
	}
	
	if(!lua_iscfunction(L, 1))
	{
		return luaL_error(L, "help expects zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	if(func == l_setcelldims)
	{
		lua_pushstring(L, "Set the dimension of each Rectangular Prism");
		lua_pushstring(L, "1 *3Vector*: The x, y and z lengths of the prism");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_getcelldims)
	{
		lua_pushstring(L, "Get the dimension of each Rectangular Prism");
		lua_pushstring(L, "");
		lua_pushstring(L, "3 Numbers: The x, y and z lengths of the prism");
		return 3;
	}
	
	if(func == l_setcrossover)
	{
		lua_pushstring(L, "Set the relative error to define the crossover from magnetostatics to dipole calculations in the interaction matrix generation. Initial value is 0.0001.");
		lua_pushstring(L, "1 Number: The relative error for the crossover");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_getcrossover)
	{
		lua_pushstring(L, "Get the relative error to define the crossover from magnetostatics to dipole calculations in the interaction matrix generation. Initial value is 0.0001.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: The relative error for the crossover");
		return 3;
	}
	
	return LongRange::help(L);
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Magnetostatic::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LongRange::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"setCellDimensions", l_setcelldims},
		{"cellDimensions",    l_getcelldims},
		{"setCrossoverTolerance", l_setcrossover},
		{"crossoverTolerance", l_setcrossover},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
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
	luaT_register<Magnetostatic>(L);
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




