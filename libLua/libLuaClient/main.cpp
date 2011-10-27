#include <iostream>
using namespace std;

#include "libLuaClient.h"

int main(int argc, char** argv)
{
	lua_State *L = lua_open();
	luaL_openlibs(L);
	registerLuaClient(L);

	if(argc != 2)
	{
		cerr << "please supply a lua script" << endl;
	}
	else
	{
		if(luaL_dofile(L, argv[1]))
			cerr << lua_tostring(L, -1) << endl;
	}

		
	lua_close(L);

	return 0;
}
