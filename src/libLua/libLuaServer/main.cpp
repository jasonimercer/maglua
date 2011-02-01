#include <iostream>
using namespace std;

#include "libLuaServer.h"

int l_func(lua_State* L)
{
	lua_pushstring(L, "server");
	return 1;
}

int l_registerCustom(lua_State* L)
{
	lua_pushcfunction(L, l_func);
	lua_setglobal(L, "func");
}

int main(int argc, char** argv)
{
	LuaServer.init(argc, argv);
	LuaServer.registerCallback = l_registerCustom;
	LuaServer.serve();

	return 0;
}
