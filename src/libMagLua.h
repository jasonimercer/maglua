extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)
#endif


void  libMagLuaArgs(int argc, char** argv, lua_State* L, int sub_process, int force_quiet);
void  libMagLua(lua_State* L, int sub_process, int force_quiet);
