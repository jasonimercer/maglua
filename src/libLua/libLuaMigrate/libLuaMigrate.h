#include <stdio.h>
#include <stdlib.h>
#include <string>

#ifndef JLUAVAR
#define JLUAVAR
struct lua_Variable
{
	int type;

	char* chunk;
	int   chunksize;
	int   chunklength;

	lua_Variable* listKey;
	lua_Variable* listVal;
	int           listlength;
};
#endif

extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>

void initLuaVariable(lua_Variable* v);
void freeLuaVariable(lua_Variable* v);
void exportLuaVariable(lua_State* L, int index, lua_Variable* v);
void importLuaVariable(lua_State* L, lua_Variable* v);
}





