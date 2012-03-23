#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef MAGLUA_EXPORTS
  #define MAGLUA_API __declspec(dllexport)
 #else
  #define MAGLUA_API __declspec(dllimport)
 #endif
#else
 #define MAGLUA_API 
#endif

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

void MagLua_set_and_run(lua_State* L, int sub_process);

