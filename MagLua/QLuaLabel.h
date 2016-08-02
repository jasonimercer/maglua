#ifndef QLUALABEL_H
#define QLUALABEL_H

#include <QLabel>
#include "QLuaWidget.h"

class QLuaLabel : public QLuaWidget
{
public:
	QLuaLabel(lua_State* L, QLabel* w) : QLuaWidget(L, w) {}
};


int lua_islabel(lua_State* L, int idx);
QLabel* lua_tolabel(lua_State* L, int idx);
QLuaLabel* lua_tolualabel(lua_State* L, int idx);
void lua_pushlualabel(lua_State* L, QLuaLabel* s);
void lua_registerlabel(lua_State* L);


#endif // QLUATABWIDGET_H
