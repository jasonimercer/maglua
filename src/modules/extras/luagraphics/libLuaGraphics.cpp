extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}

#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>


#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef LUAGRAPHICS_EXPORTS
  #define LUAGRAPHICS_API __declspec(dllexport)
 #else
  #define LUAGRAPHICS_API __declspec(dllimport)
 #endif
#else
 #define LUAGRAPHICS_API 
#endif




#include "info.h"

#include "Draw.h"
#include "DrawPOVRay.h"

#include "AABB.h"
#include "Camera.h"
#include "Color.h"
#include "Light.h"
#include "Matrix.h"
#include "Ray.h"
#include "Sphere.h"
#include "Tube.h"
#include "Vector.h"
#include "Volume.h"
#include "Group.h"



extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
        
LUAGRAPHICS_API int lib_register(lua_State* L);
LUAGRAPHICS_API int lib_version(lua_State* L);
LUAGRAPHICS_API const char* lib_name(lua_State* L);
LUAGRAPHICS_API int lib_main(lua_State* L);
}

#include <stdio.h>
LUAGRAPHICS_API int lib_register(lua_State* L)
{
	luaT_register<AABB>(L);
 	luaT_register<Camera>(L);
	luaT_register<Color>(L);
	luaT_register<Light>(L);
	luaT_register<Matrix>(L);
	luaT_register<Ray>(L);
	luaT_register<Sphere>(L);
	luaT_register<Tube>(L);
	luaT_register<Vector>(L);
	luaT_register<Volume>(L);
	luaT_register<Group>(L);
	luaT_register<Draw>(L);
	luaT_register<DrawPOVRay>(L);
	return 0;
}

LUAGRAPHICS_API int lib_version(lua_State* L)
{
	return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "LuaGraphics";
#else
	return "LuaGraphics-Debug";
#endif
}

int lib_main(lua_State* L)
{
	return 0;
}
