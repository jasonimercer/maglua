#ifndef QLUASETTINGS_H
#define QLUASETTINGS_H

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

void lua_registerqluasettings(lua_State* L);

#endif // QLUASETTINGS_H
