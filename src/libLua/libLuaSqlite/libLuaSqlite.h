#include <sqlite3.h>

extern "C" {
        #include <lua.h>
        #include <lualib.h>
        #include <lauxlib.h>
}

sqlite3* checkSQLite(lua_State* L, int index);
void lua_pushSQLite(lua_State* L, sqlite3* sql);

int registerSQLite(lua_State* L);
