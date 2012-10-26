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

#include "spinsystem.h"
#include "info.h"
#include "spinoperationmagnetostatics2d.h"


Magnetostatics2D::Magnetostatics2D(const char* Name, const int field_slot, int nx, int ny, int nz, const int encode_tag)
	: LongRange2D(Name, field_slot, nx, ny, nz, encode_tag)
{
}

Magnetostatics2D::~Magnetostatics2D()
{
}

void Magnetostatics2D::push(lua_State* L)
{
	luaT_push<Magnetostatics2D>(L, this);
}

int Magnetostatics2D::luaInit(lua_State* L)
{
	LongRange2D::luaInit(L);
	lua_getglobal(L, "Magnetostatics2D");
	lua_getfield(L, -1, "internalSetup");
	luaT_push<Magnetostatics2D>(L, this);
	lua_call(L, 1, 0);
	return 0;
}

int Magnetostatics2D::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates a Long Range Magnetostatic field for a *SpinSystem*.");
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	return LongRange2D::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Magnetostatics2D::luaMethods()
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
MAGNETOSTATICS2D_API int lib_register(lua_State* L);
MAGNETOSTATICS2D_API int lib_version(lua_State* L);
MAGNETOSTATICS2D_API const char* lib_name(lua_State* L);
MAGNETOSTATICS2D_API int lib_main(lua_State* L);
}

#include "mag2d_luafuncs.h"

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
 \
	double p[2][3]; \
	int q[2] = {4,5}; \
	for(int i=0; i<2; i++) \
	{ \
		for(int j=0; j<3; j++) \
		{ \
			lua_pushinteger(L, 1+j); \
			lua_gettable(L, q[i]); \
			p[i][j] = lua_tonumber(L, -1); \
			lua_pop(L, 1); \
		} \
	} \
	lua_pushnumber(L, F(X, Y, Z, p[0], p[1])); \
	return 1; \

static int l_xx(lua_State* L){ ___(magnetostatic_Nxx) }
static int l_xy(lua_State* L){ ___(magnetostatic_Nxy) }
static int l_xz(lua_State* L){ ___(magnetostatic_Nxz) }

static int l_yx(lua_State* L){ ___(magnetostatic_Nyx) }
static int l_yy(lua_State* L){ ___(magnetostatic_Nyy) }
static int l_yz(lua_State* L){ ___(magnetostatic_Nyz) }

static int l_zx(lua_State* L){ ___(magnetostatic_Nzx) }
static int l_zy(lua_State* L){ ___(magnetostatic_Nzy) }
static int l_zz(lua_State* L){ ___(magnetostatic_Nzz) }

MAGNETOSTATICS2D_API int lib_register(lua_State* L)
{
	luaT_register<Magnetostatics2D>(L);

	lua_getglobal(L, "Magnetostatics2D");
	
	lua_pushcfunction(L, l_xx);	lua_setfield(L, -2, "NXX");
	lua_pushcfunction(L, l_xy);	lua_setfield(L, -2, "NXY");
	lua_pushcfunction(L, l_xz);	lua_setfield(L, -2, "NXZ");

	lua_pushcfunction(L, l_yx);	lua_setfield(L, -2, "NYX");
	lua_pushcfunction(L, l_yy);	lua_setfield(L, -2, "NYY");
	lua_pushcfunction(L, l_yz);	lua_setfield(L, -2, "NYZ");

	lua_pushcfunction(L, l_zx);	lua_setfield(L, -2, "NZX");
	lua_pushcfunction(L, l_zy);	lua_setfield(L, -2, "NZY");
	lua_pushcfunction(L, l_zz);	lua_setfield(L, -2, "NZZ");
	lua_pop(L, 1); //pop table
	
	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");
	if(luaL_dostring(L, __mag2d_luafuncs()))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return luaL_error(L, lua_tostring(L, -1));
	}

	lua_pushnil(L);
	lua_setglobal(L, "maglua_getmetatable");

	return 0;
}


MAGNETOSTATICS2D_API int lib_version(lua_State* L)
{
	return __revi;
}

MAGNETOSTATICS2D_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Magnetostatics2D";
#else
	return "Magnetostatics2D-Debug";
#endif
}

MAGNETOSTATICS2D_API int lib_main(lua_State* L)
{
	return 0;
}



