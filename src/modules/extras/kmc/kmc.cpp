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

#include <math.h>
#include "kmc.h"
#include "spinsystem.h"
#include "spinoperation.h"

KMC::KMC()
	: LuaBaseObject( hash32(KMC::typeName()) )
{
	data_ref = LUA_REFNIL;
}

KMC::~KMC()
{
	deinit();
}

void KMC::deinit()
{
	luaL_unref(L, LUA_REGISTRYINDEX, data_ref);
}


int KMC::getInternalData(lua_State* L)
{
	lua_rawgeti(L, LUA_REGISTRYINDEX, data_ref);
	return 1;
}

void KMC::setInternalData(lua_State* L, int stack_pos)
{
	lua_pushvalue(L, stack_pos);
	data_ref = luaL_ref(L, LUA_REGISTRYINDEX);
}


int KMC::luaInit(lua_State* L)
{
	return LuaBaseObject::luaInit(L);
}

void KMC::encode(buffer* b)
{
	ENCODE_PREAMBLE
}

int  KMC::decode(buffer* b)
{
	return 0;
}


int KMC::help(lua_State* L)
{
    if(lua_gettop(L) == 0)
    {
        lua_pushstring(L, "Kinetic Monte Carlo operator. See ref #reference. This is the base class and serves as the framework for all KMC algorithms in MagLua. Variants are implemented with custom event functions.");
        lua_pushstring(L, "");
        lua_pushstring(L, ""); //output, empty
        return 3;
    }

	return 0;//LuaBaseObject::help(L);
}

static int l_getinternaldata(lua_State* L)
{
	LUA_PREAMBLE(KMC, kmc, 1);
	return kmc->getInternalData(L);
}
static int l_setinternaldata(lua_State* L)
{
	LUA_PREAMBLE(KMC, kmc, 1);
	kmc->setInternalData(L, 2);
	return 0;
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* KMC::luaMethods()
{
	if(m[127].name)
		return m;

	static const luaL_Reg _m[] =
    {
		{"_getinternaldata", l_getinternaldata},
		{"_setinternaldata", l_setinternaldata},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}





#include "info.h"
extern "C"
{
	KMC_API int lib_register(lua_State* L);
	KMC_API int lib_version(lua_State* L);
	KMC_API const char* lib_name(lua_State* L);
	KMC_API int lib_main(lua_State* L);
}

#include "kmc_luafuncs.h"

static int l_getmetatable(lua_State* L)
{
    if(!lua_isstring(L, 1))
        return luaL_error(L, "First argument must be a metatable name");
    luaL_getmetatable(L, lua_tostring(L, 1));
    return 1;
}

#include "kmc_wood_luafuncs.h"
KMC_API int lib_register(lua_State* L)
{
    luaT_register<KMC>(L);

	// augmenting metatable with custom lua code
    lua_pushcfunction(L, l_getmetatable);
    lua_setglobal(L, "maglua_getmetatable");

    luaL_dofile_kmc_luafuncs(L);

/*
    const char* s = __kmc_luafuncs();

    if(luaL_dostringn(L, s, "kmc_luafuncs.lua"))
    {
	fprintf(stderr, "KMC: %s\n", lua_tostring(L, -1));
	return luaL_error(L, lua_tostring(L, -1));
    }
*/

/*
    if(luaL_dostringn(L, __kmc_wood_luafuncs(), "kmc_wood_luafuncs.lua"))
    {
	fprintf(stderr, "KMC: %s\n", lua_tostring(L, -1));
	return luaL_error(L, lua_tostring(L, -1));
    }
*/

    lua_pushnil(L);
    lua_setglobal(L, "maglua_getmetatable");

    return 0;
}

KMC_API int lib_version(lua_State* L)
{
	return __revi;
}

KMC_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "KMC";
#else
	return "KMC-Debug";
#endif
}

KMC_API int lib_main(lua_State* L)
{
	return 0;
}


