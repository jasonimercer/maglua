
#include <luabaseobject.h>
#include "info.h"
extern "C"
{
int lib_register(lua_State* L);
int lib_version(lua_State* L);
const char* lib_name(lua_State* L);
int lib_main(lua_State* L);
}




#include "interactive_luafuncs.h"

int lib_register(lua_State* L)
{
    const char* s = __interactive_luafuncs();
    
    if(luaL_dostringn(L, s, "interactive_luafuncs.lua"))
    {
	fprintf(stderr, "Interactive: %s\n", lua_tostring(L, -1));
	return luaL_error(L, lua_tostring(L, -1));
    }
    return 0;
}

int lib_version(lua_State* L)
{
	return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Interactive";
#else
	return "Interactive-Debug";
#endif
}

int lib_main(lua_State* L)
{
	return 0;
}
