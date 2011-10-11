#ifndef QLUAMENU_H
#define QLUAMENU_H

#include "QLuaWidget.h"
#include <QMenu>

class QLuaMenu : public QLuaWidget
{
    Q_OBJECT
public:
    explicit QLuaMenu(QObject *parent = 0);
	explicit QLuaMenu(lua_State* L, QMenu* w) : QLuaWidget(L, w) {}

signals:

public slots:

};

#endif // QLUAMENU_H

int lua_ismenu(lua_State* L, int idx);
QLuaMenu* lua_toluamenu(lua_State* L, int idx);
void lua_pushluamenu(lua_State* L, QLuaMenu* s);
void lua_pushmenu(lua_State* L, QMenu* s);
void lua_registermenu(lua_State* L);


