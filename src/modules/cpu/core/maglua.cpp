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

#include "maglua.h"
#include "info.h"
#include "array.h"
#include "spinsystem.h"
extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
        
CORE_API int lib_register(lua_State* L);
CORE_API int lib_version(lua_State* L);
CORE_API const char* lib_name(lua_State* L);
CORE_API int lib_main(lua_State* L);
}

static int l_getmetatable(lua_State* L)
{
    if(!lua_isstring(L, 1))
        return luaL_error(L, "First argument must be a metatable name");
    luaL_getmetatable(L, lua_tostring(L, 1));
    return 1;
}


// moving getName into C. 
// profiler shows a lot of calls to it, trying to save some time
static int l_getname(lua_State* L)
{
    const int t = lua_type(L, 1);

    if(t == LUA_TSTRING)
    {
        lua_pushvalue(L, 1);
        return 1;
    }

    if(t == LUA_TNIL)
    {
        return 0;
    }

    if(lua_getmetatable(L, 1))
    {
        lua_getfield(L, -1, "slotName");
        if(lua_isfunction(L, -1))
        {
            lua_pushvalue(L, 1);
            lua_call(L, 1, 1);
            return 1;
        }
    }

    return luaL_error(L, "Unknown object used for slot name");
}


#include "spinsystem_luafuncs.h"
CORE_API int lib_register(lua_State* L)
{
	luaT_register<SpinSystem>(L);

        lua_pushcfunction(L, l_getname);
        lua_setglobal(L, "_getName");

	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");

        luaL_dofile_spinsystem_luafuncs(L);

	lua_pushnil(L);
	lua_setglobal(L, "maglua_getmetatable");

	return 0;
}

CORE_API int lib_version(lua_State* L)
{
	return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Core";
#else
	return "Core-Debug";
#endif
}

int lib_main(lua_State* L)
{
	return 0;
}
