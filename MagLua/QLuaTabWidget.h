#ifndef QLUATABWIDGET_H
#define QLUATABWIDGET_H

#include <QTabWidget>
#include "QLuaWidget.h"

class QLuaTabWidget : public QLuaWidget
{
public:
	QLuaTabWidget(lua_State* L, QTabWidget* w) : QLuaWidget(L, w) {}

	int addTab(QLuaWidget* lw, QString title);
};


int lua_istabwidget(lua_State* L, int idx);
QLuaTabWidget* lua_totabwidget(lua_State* L, int idx);
void lua_pushluatabwidget(lua_State* L, QLuaTabWidget* s);
void lua_registertabwidget(lua_State* L);


#endif // QLUATABWIDGET_H
