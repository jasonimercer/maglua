#include <luabaseobject.h>
#include "profile_main.h"
#include "profile_close.h"

#include <unistd.h>
#include <sys/types.h>

#include "info.h"

// these are also available in os_extensions but there's no guarantee that it'll 
// be loaded first or even available
static int profile_data_ref = LUA_REFNIL;

// Here's an odd bit of code. I hope the compiler isn't too smart.
// The idea here is to add a dependancy. We want the Timer module
// to load before this modules so we can start the main timer if
// it turns out we are profiling. 
#include "../../common/timer/mtimer.h"
Timer t;

static int l_getpid(lua_State* L)
{
    lua_pushinteger(L, getpid());
    return 1;
}

static int l_gethostname(lua_State* L)
{
    char n[256];
    gethostname(n, 256);
    lua_pushstring(L, n);
    return 1;
}

static int l_set_profile_data(lua_State* L)
{
    luaL_unref(L, LUA_REGISTRYINDEX, profile_data_ref);
    profile_data_ref = luaL_ref(L, LUA_REGISTRYINDEX);
    return 0;
}

static int l_get_profile_data(lua_State* L)
{
    lua_rawgeti(L, LUA_REGISTRYINDEX, profile_data_ref);
    return 1;
}



#include "info.h"
extern "C"
{
int lib_register(lua_State* L);
int lib_version(lua_State* L);
const char* lib_name(lua_State* L);
int lib_main(lua_State* L);
int lib_close(lua_State* L);
}

int lib_register(lua_State* L)
{
    return 0;
}

int lib_version(lua_State* L)
{
	return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Profile";
#else
	return "Profile-Debug";
#endif
}

#include "profile_main.h"
int lib_main(lua_State* L)
{
/*
    lua_pushcfunction(L, l_getpid);
    lua_setglobal(L, "_profile_getpid");

    lua_pushcfunction(L, l_gethostname);
    lua_setglobal(L, "_profile_gethostname");
*/
    lua_pushcfunction(L, l_set_profile_data);
    lua_setglobal(L, "_set_profile_data");
    
    luaL_dofile_profile_main(L);

    return 0;
}

int lib_close(lua_State* L)
{
    lua_pushcfunction(L, l_get_profile_data);
    lua_setglobal(L, "_get_profile_data");

    luaL_dofile_profile_close(L);
    return 0;
}
