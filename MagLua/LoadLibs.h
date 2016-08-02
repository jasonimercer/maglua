extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#include <QStringList>

int load_libs(lua_State* L, const QStringList& libs, QStringList& failList);
