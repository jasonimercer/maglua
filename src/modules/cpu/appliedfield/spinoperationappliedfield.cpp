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
	dArray& hx = (*ss->hx[slot]);
	dArray& hy = (*ss->hy[slot]);
	dArray& hz = (*ss->hz[slot]);

	const int nxyz = ss->nx * ss->ny * ss->nz; //these are not imported
	
	for(int i=0; i<nxyz; i++)
	{
		hx[i] = B[0] * global_scale;
		hy[i] = B[1] * global_scale;
		hz[i] = B[2] * global_scale;
		
		//printf("%g %g %g\n", hx[i], hy[i], hz[i]);
	}
	return true;
}




// generete canned functions for these simple get/set cases
LUAFUNC_SET_DOUBLE(AppliedField, B[0], l_sx)
LUAFUNC_SET_DOUBLE(AppliedField, B[1], l_sy)
LUAFUNC_SET_DOUBLE(AppliedField, B[2], l_sz)

LUAFUNC_GET_DOUBLE(AppliedField, B[0], l_gx)
LUAFUNC_GET_DOUBLE(AppliedField, B[1], l_gy)
LUAFUNC_GET_DOUBLE(AppliedField, B[2], l_gz)

static int l_set(lua_State* L)
{
	LUA_PREAMBLE(AppliedField, ap, 1);
	
	double a[3];
	int r = lua_getNdouble(L, 3, a, 2, 0);
	if(r<0)
		return luaL_error(L, "invalid field");
	
	ap->B[0] = a[0];
	ap->B[1] = a[1];
	ap->B[2] = a[2];
	
	return 0;
}

static int l_add(lua_State* L)
{
	LUA_PREAMBLE(AppliedField, ap, 1);
	
	double a[3];
	int r = lua_getNdouble(L, 3, a, 2, 0);
	if(r<0)
		return luaL_error(L, "invalid field");
	
	ap->B[0] += a[0];
	ap->B[1] += a[1];
	ap->B[2] += a[2];
	
	return 0;
}



static int l_get(lua_State* L)
{
	LUA_PREAMBLE(AppliedField, ap, 1);
	for(int i=0; i<3; i++)
		lua_pushnumber(L, ap->B[i]);
	return 3;
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
	
	if(!lua_iscfunction(L, 1))
	{
		return luaL_error(L, "help expect zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
		
	if(func == l_set)
	{
		lua_pushstring(L, "Set the direction and strength of the Applied Field");
		lua_pushstring(L, "1 *3Vector*: The *3Vector* defines the strength and direction of the applied field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_add)
	{
		lua_pushstring(L, "Add the direction and strength of the Applied Field");
		lua_pushstring(L, "1 *3Vector*: The *3Vector* defines the strength and direction of the applied field addition");
		lua_pushstring(L, "");
		return 3;
	}
	
	
	if(func == l_get)
	{
		lua_pushstring(L, "Get the direction and strength of the Applied Field");
		lua_pushstring(L, "");
		lua_pushstring(L, "3 numbers: The x, y and z components of the field");
		return 3;
	}
	
	if(func == l_gx)
	{
		lua_pushstring(L, "Get the X component of the applied field.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: X component of the applied field");
		return 3;
	}
	if(func == l_gy)
	{
		lua_pushstring(L, "Get the Y component of the applied field.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: Y component of the applied field");
		return 3;
	}
	if(func == l_gz)
	{
		lua_pushstring(L, "Get the Z component of the applied field.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: Z component of the applied field");
		return 3;
	}
	

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
		{"set",          l_set},
		{"get",          l_get},
		{"add",          l_add},
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

APPLIEDFIELD_API int lib_register(lua_State* L)
{
	luaT_register<AppliedField>(L);
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



