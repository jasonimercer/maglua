#ifndef QLUAACTION_H
#define QLUAACTION_H

#include "QLuaWidget.h"
#include <QAction>

class QLuaAction : public QObject
{
    Q_OBJECT
public:
	explicit QLuaAction(lua_State* L, QAction* act);
	~QLuaAction();

	void setTriggerFunction(int ref);

	QAction* action;
	lua_State* L;
	int triggerref;
	int refcount;
signals:

public slots:
	void triggered(bool b);
	void triggered();
};

#endif // QLUAACTION_H


void lua_registeraction(lua_State* L);
int lua_isaction(lua_State* L, int idx);
QLuaAction* lua_toluaaction(lua_State* L, int idx);
QAction* lua_toaction(lua_State* L, int idx);
void lua_pushluaaction(lua_State* L, QLuaAction* c);
void lua_pushaction(lua_State* L, QAction* c);
