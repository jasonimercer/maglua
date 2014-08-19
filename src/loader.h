extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

int lua_loadfile(lua_State* L);
int lua_unloadfile(lua_State* L);
