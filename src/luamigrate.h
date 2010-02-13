extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}

char* exportLuaVariable(lua_State* L, int index,   int* chunksize);
int   importLuaVariable(lua_State* L, char* chunk, int  chunksize);
