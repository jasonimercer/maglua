/******************************************************************************
* Copyright (C) 2008-2013 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#include "spinoperationappliedfield.h"
#include "spinsystem.h"

#include <stdlib.h>

AppliedField::AppliedField(int nx, int ny, int nz)
	: SpinOperation(nx, ny, nz, hash32(AppliedField::typeName()))
{
	setSlotName("Zeeman");
	B[0] = 0;
	B[1] = 0;
	B[2] = 0;
}

int AppliedField::luaInit(lua_State* L)
{
	return SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
}

	
void AppliedField::encode(buffer* b)
{
	ENCODE_PREAMBLE
	SpinOperation::encode(b);
	char version = 0;
	encodeChar(version, b);
	encodeDouble(B[0], b);
	encodeDouble(B[1], b);
	encodeDouble(B[2], b);
}

int  AppliedField::decode(buffer* b)
{
	SpinOperation::decode(b);
	char version = decodeChar(b);
	if(version == 0)
	{
		B[0] = decodeDouble(b);
		B[1] = decodeDouble(b);
		B[2] = decodeDouble(b);
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}
	return 0;
}

AppliedField::~AppliedField()
{
}

bool AppliedField::apply(SpinSystem** ss, int n)
{
	for(int i=0; i<n; i++)
		apply(ss[i]);
	return true;
}

bool AppliedField::apply(SpinSystem* ss)
{
	// the following is implemented in appliedfield_luafuncs.lua
	// leaving this here for now in case people want to make direct 
	// calls to the C implementation.
	int slot = markSlotUsed(ss);

	ss->hx[slot]->addValue(B[0]*global_scale);
	ss->hy[slot]->addValue(B[1]*global_scale);
	ss->hz[slot]->addValue(B[2]*global_scale);
	return true;
}

static int l_sx(lua_State* L)
{
	LUA_PREAMBLE(AppliedField, af, 1);
	af->B[0] = lua_tonumber(L, 2);
	return 0;
}
static int l_sy(lua_State* L)
{
	LUA_PREAMBLE(AppliedField, af, 1);
	af->B[1] = lua_tonumber(L, 2);
	return 0;
}
static int l_sz(lua_State* L)
{
	LUA_PREAMBLE(AppliedField, af, 1);
	af->B[2] = lua_tonumber(L, 2);
	return 0;
}

static int l_gx(lua_State* L)
{
	LUA_PREAMBLE(AppliedField, af, 1);
	lua_pushnumber(L, af->B[0]);
	return 1;
}
static int l_gy(lua_State* L)
{
	LUA_PREAMBLE(AppliedField, af, 1);
	lua_pushnumber(L, af->B[1]);
	return 1;
}
static int l_gz(lua_State* L)
{
	LUA_PREAMBLE(AppliedField, af, 1);
	lua_pushnumber(L, af->B[2]);
	return 1;
}

int AppliedField::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Applies an external, global field to a *SpinSystem*");
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
#define GETHELP(f, C) \
	if(func == f) \
	{ \
		lua_pushstring(L, "Get the " C " component of the applied field."); \
		lua_pushstring(L, ""); \
		lua_pushstring(L, "1 Number: " C " component of the applied field"); \
		return 3; \
	}
	
	GETHELP(l_gx, "X")
	GETHELP(l_gy, "Y")
	GETHELP(l_gz, "Z")
	
	

	if(func == l_sx)
	{
		lua_pushstring(L, "set the X component of the applied field.");
		lua_pushstring(L, "1 Number: X component of the applied field");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_sy)
	{
		lua_pushstring(L, "set the Y component of the applied field.");
		lua_pushstring(L, "1 Number: Y component of the applied field");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_sz)
	{
		lua_pushstring(L, "set the Z component of the applied field.");
		lua_pushstring(L, "1 Number: Z component of the applied field");
		lua_pushstring(L, "");
		return 3;
	}

	return SpinOperation::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* AppliedField::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, SpinOperation::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"x",            l_gx},
		{"y",            l_gy},
		{"z",            l_gz},
		{"setX",         l_sx},
		{"setY",         l_sy},
		{"setZ",         l_sz},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}




#include "info.h"
extern "C"
{
APPLIEDFIELD_API int lib_register(lua_State* L);
APPLIEDFIELD_API int lib_version(lua_State* L);
APPLIEDFIELD_API const char* lib_name(lua_State* L);
APPLIEDFIELD_API int lib_main(lua_State* L);
}

#include "appliedfield_luafuncs.h"
#include "appliedfield_heterogeneous_luafuncs.h"
#include "appliedfield_site_luafuncs.h"

static int l_getmetatable(lua_State* L)
{
    if(!lua_isstring(L, 1))
        return luaL_error(L, "First argument must be a metatable name");
    luaL_getmetatable(L, lua_tostring(L, 1));
    return 1;
}

#include "spinoperationappliedfield_heterogeneous.h"
#include "spinoperationappliedfield_site.h"
APPLIEDFIELD_API int lib_register(lua_State* L)
{
	luaT_register<AppliedField>(L);
	luaT_register<AppliedField_Heterogeneous>(L);
	luaT_register<AppliedField_Site>(L);
	
	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");
	
	if(luaL_dostringn(L, __appliedfield_luafuncs(), "appliedfield_luafuncs.lua"))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return luaL_error(L, lua_tostring(L, -1));
	}

	if(luaL_dostringn(L, __appliedfield_heterogeneous_luafuncs(), "appliedfield_heterogeneous_luafuncs.lua"))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return luaL_error(L, lua_tostring(L, -1));
	}

	if(luaL_dostringn(L, __appliedfield_site_luafuncs(), "appliedfield_site_luafuncs.lua"))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return luaL_error(L, lua_tostring(L, -1));
	}

	lua_pushnil(L);
	lua_setglobal(L, "maglua_getmetatable");
	
	return 0;
}


APPLIEDFIELD_API int lib_version(lua_State* L)
{
	return __revi;
}

APPLIEDFIELD_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "AppliedField";
#else
	return "AppliedField-Debug";
#endif
}

APPLIEDFIELD_API int lib_main(lua_State* L)
{
	return 0;
}



