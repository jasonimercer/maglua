#ifndef CLASSES_H
#define CLASSES_H

#include <QWidget>
class QLuaWidget;
extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}


QWidget* lua_towidget(lua_State* L, int idx);
QLuaWidget* lua_toluawidget(lua_State* L, int idx);
int lua_isluawidget(lua_State* L, int idx);

int lua_registerwidgets(lua_State* L);


#endif // CLASSES_H
