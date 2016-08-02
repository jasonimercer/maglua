#ifndef QLUAWIDGET_H
#define QLUAWIDGET_H

#include <QObject>
#include <QWidget>
#include <QList>
//#include "MainWindow.h"

#include <stdio.h>

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

class QLuaAction;

class QLuaWidget : public QObject
{
public:
	explicit QLuaWidget(lua_State* _L, QWidget* _widget);
	~QLuaWidget();

	lua_State* L;
	QWidget* widget;
	int refcount;

	// manages refcounts
	void addChild(QLuaWidget* w);
	void removeChild(QLuaWidget* w);

	void addChild(QLuaAction* w);
	void removeChild(QLuaAction* w);

	QList<QLuaWidget*> children;
	QList<QLuaAction*> children_actions;

	virtual void setFocus();
};

#endif // QLUAWIDGET_H
