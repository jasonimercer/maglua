#include "math_vectors.h"
#include <math.h>
#include "math_vectors_luafuncs.h"

#include "info.h"
extern "C"
{
MATH_VECTORS_API int lib_register(lua_State* L);
MATH_VECTORS_API int lib_version(lua_State* L);
MATH_VECTORS_API const char* lib_name(lua_State* L);
MATH_VECTORS_API int lib_main(lua_State* L);
}


static int l_erfc(lua_State* L)
{
	if(lua_isnumber(L, 1))
	{
		double x = lua_tonumber(L, 1);
		lua_pushnumber(L, erfc(x));
		return 1;
	}
	return 0;
}
static int l_erf(lua_State* L)
{
	if(lua_isnumber(L, 1))
	{
		double x = lua_tonumber(L, 1);
		lua_pushnumber(L, erf(x));
		return 1;
	}
	return 0;
}

MATH_VECTORS_API int lib_register(lua_State* L)
{
	lua_getglobal(L, "math");
	lua_pushstring(L, "erfc");
	lua_pushcfunction(L, l_erfc);
	lua_settable(L, -3);
	lua_pushstring(L, "erf");
	lua_pushcfunction(L, l_erf);
	lua_settable(L, -3);
	lua_pop(L, 1); // math table

	if(luaL_dostring(L, __math_vectors_luafuncs()))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return luaL_error(L, lua_tostring(L, -1));
	}
	
	return 0;
}

MATH_VECTORS_API int lib_version(lua_State* L)
{
	return __revi;
}

MATH_VECTORS_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Math-Vectors";
#else
	return "Math-Vectors-Debug";
#endif
}

MATH_VECTORS_API int lib_main(lua_State* L)
{
	return 0;
}
