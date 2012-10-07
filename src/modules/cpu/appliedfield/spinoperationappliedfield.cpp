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

#include "spinoperationappliedfield.h"
#include "spinsystem.h"

#include <stdlib.h>

AppliedField::AppliedField(int nx, int ny, int nz)
	: SpinOperation(AppliedField::typeName(), APPLIEDFIELD_SLOT, nx, ny, nz, hash32(AppliedField::typeName()))
{
	B[0] = 0;
	B[1] = 0;
	B[2] = 0;
}

int AppliedField::luaInit(lua_State* L)
{
	return SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
}

void AppliedField::push(lua_State* L)
{
	luaT_push<AppliedField>(L, this);
}
	
	
void AppliedField::encode(buffer* b)
{
	SpinOperation::encode(b);
	encodeDouble(B[0], b);
	encodeDouble(B[1], b);
	encodeDouble(B[2], b);
}

int  AppliedField::decode(buffer* b)
{
	SpinOperation::decode(b);
	B[0] = decodeDouble(b);
	B[1] = decodeDouble(b);
	B[2] = decodeDouble(b);
	return 0;
}

AppliedField::~AppliedField()
{
}

bool AppliedField::apply(SpinSystem* ss)
{
	// the following is implemented in appliedfield_luafuncs.lua
	// leaving this here for now in case people want to make direct 
	// calls to the C implementation.
	markSlotUsed(ss);
	ss->ensureSlotExists(slot);

	ss->hx[slot]->setAll(B[0]*global_scale);
	ss->hy[slot]->setAll(B[1]*global_scale);
	ss->hz[slot]->setAll(B[2]*global_scale);
	return true;
}


// generete canned functions for these simple get/set cases
LUAFUNC_SET_DOUBLE(AppliedField, B[0], l_sx)
LUAFUNC_SET_DOUBLE(AppliedField, B[1], l_sy)
LUAFUNC_SET_DOUBLE(AppliedField, B[2], l_sz)

LUAFUNC_GET_DOUBLE(AppliedField, B[0], l_gx)
LUAFUNC_GET_DOUBLE(AppliedField, B[1], l_gy)
LUAFUNC_GET_DOUBLE(AppliedField, B[2], l_gz)


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
static int l_getmetatable(lua_State* L)
{
    if(!lua_isstring(L, 1))
        return luaL_error(L, "First argument must be a metatable name");
    luaL_getmetatable(L, lua_tostring(L, 1));
    return 1;
}

APPLIEDFIELD_API int lib_register(lua_State* L)
{
	luaT_register<AppliedField>(L);
	
	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");
	if(luaL_dostring(L, __appliedfield_luafuncs()))
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



