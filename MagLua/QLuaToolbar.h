#ifndef QLUATOOLBAR_H
#define QLUATOOLBAR_H

#include <QToolBar>
#include "QLuaWidget.h"
#include "QLuaAction.h"

class QLuaToolBar : public QLuaWidget
{
    Q_OBJECT
public:
	explicit QLuaToolBar(lua_State* L, QToolBar *tb);
	~QLuaToolBar();
signals:

public slots:

};

#endif // QLUATOOLBAR_H

void lua_registertoolbar(lua_State* L);
void lua_pushluatoolbar(lua_State* L, QLuaToolBar* c);
QToolBar* lua_totoolbar(lua_State* L, int idx);
QLuaToolBar* lua_toluatoolbar(lua_State* L, int idx);
int lua_istoolbar(lua_State* L, int idx);

