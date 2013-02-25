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
#ifndef WIN32
#include <strings.h>
#endif

#include <stdlib.h>
#include <math.h>

Dipole::Dipole(int nx, int ny, int nz)
	: LongRange(Dipole::typeName(), DIPOLE_SLOT, nx, ny, nz, hash32(Dipole::typeName()))
{

}

int Dipole::luaInit(lua_State* L)
{
	return LongRange::luaInit(L); //gets nx, ny, nz, nxyz
}

void Dipole::encode(buffer* b)
{
	LongRange::encode(b);
}

int  Dipole::decode(buffer* b)
{
	deinit();
	LongRange::decode(b);
	
	return 0;
}

Dipole::~Dipole()
{
	deinit();
}

void Dipole::loadMatrixFunction(double* XX, double* XY, double* XZ, double* YY, double* YZ, double* ZZ)
{
	dipoleLoad(
		nx, ny, nz,
		gmax, ABC,
		XX, XY, XZ,
		YY, YZ, ZZ);
}




int Dipole::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates the dipolar field of a *SpinSystem*");
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	return LongRange::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Dipole::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LongRange::luaMethods());
	static const luaL_Reg _m[] =
	{
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}



extern "C"
{
DIPOLE_API int lib_register(lua_State* L);
DIPOLE_API int lib_version(lua_State* L);
DIPOLE_API const char* lib_name(lua_State* L);
DIPOLE_API int lib_main(lua_State* L);
}

#include "dipolesupport.h"
#define __d(F)					   \
	double X = lua_tonumber(L, 1); \
	double Y = lua_tonumber(L, 2); \
	double Z = lua_tonumber(L, 3); \
	lua_pushnumber(L, F(X, Y, Z)); \
	return 1; \
	
static int l_xx(lua_State* L){ __d(gamma_xx_dip) }
static int l_xy(lua_State* L){ __d(gamma_xy_dip) }
static int l_xz(lua_State* L){ __d(gamma_xz_dip) }

static int l_yx(lua_State* L){ __d(gamma_xy_dip) }
static int l_yy(lua_State* L){ __d(gamma_yy_dip) }
static int l_yz(lua_State* L){ __d(gamma_yz_dip) }

static int l_zx(lua_State* L){ __d(gamma_xz_dip) }
static int l_zy(lua_State* L){ __d(gamma_xy_dip) }
static int l_zz(lua_State* L){ __d(gamma_zz_dip) }

DIPOLE_API int lib_register(lua_State* L)
{
	luaT_register<Dipole>(L);

	lua_getglobal(L, "Dipole");
	
	lua_pushcfunction(L, l_xx);	lua_setfield(L, -2, "XX");
	lua_pushcfunction(L, l_xy);	lua_setfield(L, -2, "XY");
	lua_pushcfunction(L, l_xz);	lua_setfield(L, -2, "XZ");

	lua_pushcfunction(L, l_yx);	lua_setfield(L, -2, "YX");
	lua_pushcfunction(L, l_yy);	lua_setfield(L, -2, "YY");
	lua_pushcfunction(L, l_yz);	lua_setfield(L, -2, "YZ");

	lua_pushcfunction(L, l_zx);	lua_setfield(L, -2, "ZX");
	lua_pushcfunction(L, l_zy);	lua_setfield(L, -2, "ZY");
	lua_pushcfunction(L, l_zz);	lua_setfield(L, -2, "ZZ");
	lua_pop(L, 1);
	
	
	return 0;
}

DIPOLE_API int lib_version(lua_State* L)
{
	return __revi;
}


DIPOLE_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Dipole";
#else
	return "Dipole-Debug";
#endif
}

DIPOLE_API int lib_main(lua_State* L)
{
	return 0;
}


