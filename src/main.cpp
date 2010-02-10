#include "main.h"
#include "info.h"

int checkargs(int argc, char** argv)
{
	if(argc < 2)
	{
		cerr << __info << endl;
		cerr << "Please supply a Maglua script" << endl;
		return 1;
	}
	return 0;
}

void lua_addargs(lua_State* L, int argc, char** argv)
{
	lua_pushinteger(L, argc);
	lua_setglobal(L, "argc");

	lua_newtable(L);
	for(int i=0; i<argc; i++)
	{
		lua_pushinteger(L, i+1);
		lua_pushstring(L, argv[i]);
		lua_settable(L, -3);
	}
	lua_setglobal(L, "argv");
}


int main(int argc, char** argv)
{
	if(checkargs(argc, argv))
		return 1;
	
	lua_State *L = lua_open();
	luaL_openlibs(L);
	registerSpinSystem(L);
	registerLLG(L);
	registerExchange(L);
	registerAppliedField(L);
	registerAnisotropy(L);
	registerDipole(L);
	registerRandom(L);
	registerThermal(L);
	registerConvert(L);
	registerInterpolatingFunction(L);
	registerInterpolatingFunction2D(L);
	
	lua_addargs(L, argc, argv);

	if(luaL_dofile(L, argv[1]))
		cerr << lua_tostring(L, -1) << endl;
	
	lua_close(L);
	
	return 0;
}
