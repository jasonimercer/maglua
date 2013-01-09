#include "math_vectors.h"

#include "math_vectors_luafuncs.h"

#include "info.h"
extern "C"
{
MATH_VECTORS_API int lib_register(lua_State* L);
MATH_VECTORS_API int lib_version(lua_State* L);
MATH_VECTORS_API const char* lib_name(lua_State* L);
MATH_VECTORS_API int lib_main(lua_State* L);
}


MATH_VECTORS_API int lib_register(lua_State* L)
{
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
