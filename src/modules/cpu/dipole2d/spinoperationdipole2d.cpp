/******************************************************************************
* Copyright (C) 2008-2012 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#include "spinsystem.h"
#include "info.h"
#include "spinoperationdipole2d.h"


Dipole2D::Dipole2D(int nx, int ny, int nz, const int encode_tag)
	: LongRange2D(nx, ny, nz, encode_tag)
{
}

Dipole2D::~Dipole2D()
{
}

int Dipole2D::luaInit(lua_State* L)
{
	LongRange2D::luaInit(L);
	lua_getglobal(L, "Dipole2D");
	lua_getfield(L, -1, "internalSetup");
	luaT_push<Dipole2D>(L, this);
	lua_call(L, 1, 0);
	return 0;
}

int Dipole2D::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates a Long Range Dipole field for a *SpinSystem*.");
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	return LongRange2D::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Dipole2D::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LongRange2D::luaMethods());
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
DIPOLE2D_API int lib_register(lua_State* L);
DIPOLE2D_API int lib_version(lua_State* L);
DIPOLE2D_API const char* lib_name(lua_State* L);
DIPOLE2D_API int lib_main(lua_State* L);
}

#include "dip2d_luafuncs.h"

static int l_getmetatable(lua_State* L)
{
	if(!lua_isstring(L, 1))
		return luaL_error(L, "First argument must be a metatable name");
	luaL_getmetatable(L, lua_tostring(L, 1));
	return 1;
}

#include "demag.h"
#define ___(F)					   \
	double X = lua_tonumber(L, 1); \
	double Y = lua_tonumber(L, 2); \
	double Z = lua_tonumber(L, 3); \
	lua_pushnumber(L, F(X, Y, Z)); \
	return 1; \

#define __A(F)					   \
	double X = lua_tonumber(L, 1); \
	double Y = lua_tonumber(L, 2); \
	double Z = lua_tonumber(L, 3); \
	double sx = lua_tonumber(L, 4); \
	double sy = lua_tonumber(L, 5); \
	int nx = lua_tonumber(L, 6); \
	int ny = lua_tonumber(L, 7); \
 \
	lua_pushnumber(L, F(X, Y, Z, sx, sy, nx, ny));	\
	return 1; \

static int l_xx(lua_State* L){ ___(dipole2d_Nxx) }
static int l_xy(lua_State* L){ ___(dipole2d_Nxy) }
static int l_xz(lua_State* L){ ___(dipole2d_Nxz) }

static int l_yx(lua_State* L){ ___(dipole2d_Nyx) }
static int l_yy(lua_State* L){ ___(dipole2d_Nyy) }
static int l_yz(lua_State* L){ ___(dipole2d_Nyz) }

static int l_zx(lua_State* L){ ___(dipole2d_Nzx) }
static int l_zy(lua_State* L){ ___(dipole2d_Nzy) }
static int l_zz(lua_State* L){ ___(dipole2d_Nzz) }



static int l_xx_r(lua_State* L){ __A(dipole2d_Nxx_range) }
static int l_xy_r(lua_State* L){ __A(dipole2d_Nxy_range) }
static int l_xz_r(lua_State* L){ __A(dipole2d_Nxz_range) }

static int l_yx_r(lua_State* L){ __A(dipole2d_Nyx_range) }
static int l_yy_r(lua_State* L){ __A(dipole2d_Nyy_range) }
static int l_yz_r(lua_State* L){ __A(dipole2d_Nyz_range) }

static int l_zx_r(lua_State* L){ __A(dipole2d_Nzx_range) }
static int l_zy_r(lua_State* L){ __A(dipole2d_Nzy_range) }
static int l_zz_r(lua_State* L){ __A(dipole2d_Nzz_range) }

DIPOLE2D_API int lib_register(lua_State* L)
{
	luaT_register<Dipole2D>(L);

	lua_getglobal(L, "Dipole2D");
	
	lua_pushcfunction(L, l_xx);	lua_setfield(L, -2, "NXX");
	lua_pushcfunction(L, l_xy);	lua_setfield(L, -2, "NXY");
	lua_pushcfunction(L, l_xz);	lua_setfield(L, -2, "NXZ");

	lua_pushcfunction(L, l_yx);	lua_setfield(L, -2, "NYX");
	lua_pushcfunction(L, l_yy);	lua_setfield(L, -2, "NYY");
	lua_pushcfunction(L, l_yz);	lua_setfield(L, -2, "NYZ");

	lua_pushcfunction(L, l_zx);	lua_setfield(L, -2, "NZX");
	lua_pushcfunction(L, l_zy);	lua_setfield(L, -2, "NZY");
	lua_pushcfunction(L, l_zz);	lua_setfield(L, -2, "NZZ");


	lua_pushcfunction(L, l_xx_r);	lua_setfield(L, -2, "NXX_r");
	lua_pushcfunction(L, l_xy_r);	lua_setfield(L, -2, "NXY_r");
	lua_pushcfunction(L, l_xz_r);	lua_setfield(L, -2, "NXZ_r");

	lua_pushcfunction(L, l_yx_r);	lua_setfield(L, -2, "NYX_r");
	lua_pushcfunction(L, l_yy_r);	lua_setfield(L, -2, "NYY_r");
	lua_pushcfunction(L, l_yz_r);	lua_setfield(L, -2, "NYZ_r");

	lua_pushcfunction(L, l_zx_r);	lua_setfield(L, -2, "NZX_r");
	lua_pushcfunction(L, l_zy_r);	lua_setfield(L, -2, "NZY_r");
	lua_pushcfunction(L, l_zz_r);	lua_setfield(L, -2, "NZZ_r");

	lua_pop(L, 1); //pop table
	
	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");
	if(luaL_dostring(L, __dip2d_luafuncs()))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return luaL_error(L, lua_tostring(L, -1));
	}

	lua_pushnil(L);
	lua_setglobal(L, "maglua_getmetatable");

	return 0;
}


DIPOLE2D_API int lib_version(lua_State* L)
{
	return __revi;
}

DIPOLE2D_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Dipole2D";
#else
	return "Dipole2D-Debug";
#endif
}

DIPOLE2D_API int lib_main(lua_State* L)
{
	return 0;
}



