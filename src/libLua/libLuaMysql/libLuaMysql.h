#include <mysql.h>

extern "C" {
        #include <lua.h>
        #include <lualib.h>
        #include <lauxlib.h>
}

int registerMySQL(lua_State* L);

