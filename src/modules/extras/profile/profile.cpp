#include <luabaseobject.h>
#include "profile_support.h"
#include "profile_main.h"
#include "profile_close.h"
#include "profile.h"

#include <unistd.h>
#include <sys/types.h>

#include "info.h"

// these are also available in os_extensions but there's no guarantee that it'll 
// be loaded first or even available
static int profile_data_ref = LUA_REFNIL;

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

static int l_set_support(lua_State* L)
{
    profile_support_set_lookup(luaL_ref(L, LUA_REGISTRYINDEX));

    return 0;
}

static int l_profile_start(lua_State* L)
{
    profile_support_start(L);
    return 0;
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
    
    lua_pushcfunction(L, l_profile_start);
    lua_setglobal(L, "_profile_start");

    lua_pushcfunction(L, l_set_support);
    lua_setglobal(L, "_profile_set_lookup");

    luaL_dofile_profile_main(L);

    return 0;
}

int lib_close(lua_State* L)
{
    lua_pushcfunction(L, l_get_profile_data);
    lua_setglobal(L, "_get_profile_data");

    lua_pushcfunction(L, l_getpid);
    lua_setglobal(L, "_profile_getpid");

    lua_pushcfunction(L, l_gethostname);
    lua_setglobal(L, "_profile_gethostname");

    lua_pushcfunction(L, profile_support_stop);
    lua_setglobal(L, "_profile_support_stop");
    
    luaL_dofile_profile_close(L);
    return 0;
}
