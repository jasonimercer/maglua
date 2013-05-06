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
#include "spinoperationmagnetostatics3d.h"


Magnetostatics3D::Magnetostatics3D(int nx, int ny, int nz, const int encode_tag)
	: LongRange3D(nx, ny, nz, encode_tag)
{
}

Magnetostatics3D::~Magnetostatics3D()
{
}

void Magnetostatics3D::encode(buffer* b)
{
	LongRange3D::encode(b);
	char version = 0;
	encodeChar(version, b);
}

int  Magnetostatics3D::decode(buffer* b)
{
	int i = LongRange3D::decode(b);
	
	char version = decodeChar(b);
	if(version == 0)
	{
		int s = lua_gettop(L);
		lua_getglobal(L, "Magnetostatics3D");
		lua_getfield(L, -1, "internalSetup");
		luaT_push<Magnetostatics3D>(L, this);
		lua_call(L, 1, 0);
		while(lua_gettop(L) > s)
			lua_pop(L, 1);
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}

	return i;
}


// this is a base 0 function
double Magnetostatics3D::getGrainSize()
{
	// info in longrange_ref
	lua_rawgeti(L, LUA_REGISTRYINDEX, longrange_ref);
	
	lua_pushstring(L, "grainSize"); 
	lua_gettable(L, -2);// grain size table on stack
	
	double xyz[3];
	
	for(int i=0; i<3; i++)
	{
		lua_pushinteger(L, i+1);
		lua_gettable(L, -2);
		xyz[i] = lua_tonumber(L, -1);
		lua_pop(L, 1);
	}
	lua_pop(L, 2);
	
	return xyz[0] * xyz[1] * xyz[2];
	
}


int Magnetostatics3D::luaInit(lua_State* L)
{
	LongRange3D::luaInit(L);
	int s = lua_gettop(L);
	lua_getglobal(L, "Magnetostatics3D");
	lua_getfield(L, -1, "internalSetup");
	luaT_push<Magnetostatics3D>(L, this);
	lua_call(L, 1, 0);
	while(lua_gettop(L) > s)
		lua_pop(L, 1);
	return 0;
}

int Magnetostatics3D::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates a Long Range Magnetostatic field for a *SpinSystem*.");
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	return LongRange3D::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Magnetostatics3D::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LongRange3D::luaMethods());
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
MAGNETOSTATICS3D_API int lib_register(lua_State* L);
MAGNETOSTATICS3D_API int lib_version(lua_State* L);
MAGNETOSTATICS3D_API const char* lib_name(lua_State* L);
MAGNETOSTATICS3D_API int lib_main(lua_State* L);
}

#include "mag3d_luafuncs.h"

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



#include "pointfunc_demag.h"
#define __p(F)					   \
	double X = lua_tonumber(L, 1); \
	double Y = lua_tonumber(L, 2); \
	double Z = lua_tonumber(L, 3); \
 \
	double s[3]; \
	for(int j=0; j<3; j++) \
	{ \
		lua_pushinteger(L, 1+j); \
		lua_gettable(L, 4); \
		s[j] = lua_tonumber(L, -1); \
		lua_pop(L, 1); \
	} \
	lua_pushnumber(L, F(X, Y, Z, s)); \
	return 1; \
	
static int l_pxx(lua_State* L){ __p(magnetostatic_Pxx) }
static int l_pxy(lua_State* L){ __p(magnetostatic_Pxy) }
static int l_pxz(lua_State* L){ __p(magnetostatic_Pxz) }

static int l_pyx(lua_State* L){ __p(magnetostatic_Pyx) }
static int l_pyy(lua_State* L){ __p(magnetostatic_Pyy) }
static int l_pyz(lua_State* L){ __p(magnetostatic_Pyz) }

static int l_pzx(lua_State* L){ __p(magnetostatic_Pzx) }
static int l_pzy(lua_State* L){ __p(magnetostatic_Pzy) }
static int l_pzz(lua_State* L){ __p(magnetostatic_Pzz) }

MAGNETOSTATICS3D_API int lib_register(lua_State* L)
{
	luaT_register<Magnetostatics3D>(L);

	lua_getglobal(L, "Magnetostatics3D");
	
	lua_pushcfunction(L, l_xx);	lua_setfield(L, -2, "NXX");
	lua_pushcfunction(L, l_xy);	lua_setfield(L, -2, "NXY");
	lua_pushcfunction(L, l_xz);	lua_setfield(L, -2, "NXZ");

	lua_pushcfunction(L, l_yx);	lua_setfield(L, -2, "NYX");
	lua_pushcfunction(L, l_yy);	lua_setfield(L, -2, "NYY");
	lua_pushcfunction(L, l_yz);	lua_setfield(L, -2, "NYZ");

	lua_pushcfunction(L, l_zx);	lua_setfield(L, -2, "NZX");
	lua_pushcfunction(L, l_zy);	lua_setfield(L, -2, "NZY");
	lua_pushcfunction(L, l_zz);	lua_setfield(L, -2, "NZZ");
	
		
	lua_pushcfunction(L, l_pxx);	lua_setfield(L, -2, "PXX");
	lua_pushcfunction(L, l_pxy);	lua_setfield(L, -2, "PXY");
	lua_pushcfunction(L, l_pxz);	lua_setfield(L, -2, "PXZ");

	lua_pushcfunction(L, l_pyx);	lua_setfield(L, -2, "PYX");
	lua_pushcfunction(L, l_pyy);	lua_setfield(L, -2, "PYY");
	lua_pushcfunction(L, l_pyz);	lua_setfield(L, -2, "PYZ");

	lua_pushcfunction(L, l_pzx);	lua_setfield(L, -2, "PZX");
	lua_pushcfunction(L, l_pzy);	lua_setfield(L, -2, "PZY");
	lua_pushcfunction(L, l_pzz);	lua_setfield(L, -2, "PZZ");
	lua_pop(L, 1); //pop table
	
	register_mag3d_internal_functions(L);
	
	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");
	if(luaL_dostring(L, __mag3d_luafuncs()))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return luaL_error(L, lua_tostring(L, -1));
	}

	lua_pushnil(L);
	lua_setglobal(L, "maglua_getmetatable");

	return 0;
}


MAGNETOSTATICS3D_API int lib_version(lua_State* L)
{
	return __revi;
}

MAGNETOSTATICS3D_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Magnetostatics3D";
#else
	return "Magnetostatics3D-Debug";
#endif
}

MAGNETOSTATICS3D_API int lib_main(lua_State* L)
{
	return 0;
}



