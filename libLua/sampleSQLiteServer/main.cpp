#include <iostream>
using namespace std;

#include <libLuaServer.h>
#include <libLuaClient.h>
#include <libLuaSqlite.h>

int l_registerCustom(lua_State* L)
{
	registerSQLite(L);
	registerLuaClient(L);
}

int main(int argc, char** argv)
{
	LuaServer.init(argc, argv);
	LuaServer.registerCallback = l_registerCustom;
	LuaServer.serve();
	
	return 0;
}
