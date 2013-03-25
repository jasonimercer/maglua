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

void AppliedField::encode(buffer* b)
{
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
	{
		markSlotUsed(ss[i]);
		ss[i]->ensureSlotExists(slot);

		ss[i]->hx[slot]->setAll(B[0]*global_scale);
		ss[i]->hy[slot]->setAll(B[1]*global_scale);
		ss[i]->hz[slot]->setAll(B[2]*global_scale);
	}
	return true;
}


bool AppliedField::apply(SpinSystem* ss)
{
	markSlotUsed(ss);
	ss->ensureSlotExists(slot);

	ss->hx[slot]->setAll(B[0]*global_scale);
	ss->hy[slot]->setAll(B[1]*global_scale);
	ss->hz[slot]->setAll(B[2]*global_scale);
	
	return true;
}

bool AppliedField::applyToSum(SpinSystem* ss)
{
//	ss->ensureSlotExists(slot);

	ss->hx[SUM_SLOT]->addValue(B[0]*global_scale);
	ss->hy[SUM_SLOT]->addValue(B[1]*global_scale);
	ss->hz[SUM_SLOT]->addValue(B[2]*global_scale);
	
	ss->slot_used[SUM_SLOT] = true;

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
APPLIEDFIELDCUDA_API int lib_register(lua_State* L);
APPLIEDFIELDCUDA_API int lib_version(lua_State* L);
APPLIEDFIELDCUDA_API const char* lib_name(lua_State* L);
APPLIEDFIELDCUDA_API int lib_main(lua_State* L);
}

int lib_register(lua_State* L)
{
	luaT_register<AppliedField>(L);
	return 0;
}

int lib_version(lua_State* L)
{
	return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "AppliedField-Cuda";
#else
	return "AppliedField-Cuda-Debug";
#endif
}

int lib_main(lua_State* L)
{
	return 0;
}

