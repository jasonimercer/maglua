extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

typedef struct Timer Timer;

void lua_pushtimer(lua_State* L, Timer* t);
int lua_istimer(lua_State* L, int idx)
Timer* lua_totimer(lua_State* L, int idx);
void registertimer(lua_State* L);
