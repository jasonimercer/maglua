extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}


#include <string.h>
inline int luaL_dostringn(lua_State* L, const char* code, const char* name)
{
    return luaL_loadbuffer(L, code, strlen(code), name) || lua_pcall(L, 0, LUA_MULTRET, 0);
}

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)
#endif


/*
 * Returns 
 * 0 for success
 * 1 for error in script
 * 2 for error in bootloader
 */
int  libMagLuaArgs(int argc, char** argv, lua_State* L, int sub_process, int force_quiet);
int  libMagLua(lua_State* L, int sub_process, int force_quiet);
