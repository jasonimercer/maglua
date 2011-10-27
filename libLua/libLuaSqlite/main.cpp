#include <iostream>
using namespace std;

extern "C" {
        #include <lua.h>
        #include <lualib.h>
        #include <lauxlib.h>
}

#include "libLuaSqlite.h"

int pushtraceback(lua_State* L)
{
	lua_getglobal(L, "debug");
	if (!lua_istable(L, -1)) {
		lua_pop(L, 1);
		return 1;
	}
	lua_getfield(L, -1, "traceback");
	if (!lua_isfunction(L, -1)) {
		lua_pop(L, 2);
		return 1;
	}
	lua_remove(L, 1); //remove debug table
	return 0;
}

int main(int argc, char** argv)
{
	lua_State *L = lua_open();
	luaL_openlibs(L);
	registerSQLite(L);
	
	pushtraceback(L);
	if(luaL_loadfile(L, argv[1]) || lua_pcall(L, 0, LUA_MULTRET, -2))
		cerr << lua_tostring(L, -1) << endl;

	lua_close(L);

	return 0;
}