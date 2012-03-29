extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

void libMagLuaArgs(int argc, char** argv, lua_State* L, int sub_process, int force_quiet);
void libMagLua(lua_State* L, int sub_process, int force_quiet);
}
