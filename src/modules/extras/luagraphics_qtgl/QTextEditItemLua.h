#ifndef QTEXTEDITITEMLUA_H
#define QTEXTEDITITEMLUA_H

#include <QTextEdit>
#include "QLuaHilighter.h"
#include <QGraphicsProxyWidget>

#include "luabaseobject.h"
extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}

class QTextEditItemLua : public LuaBaseObject
{
public:
	QTextEditItemLua();
	~QTextEditItemLua();

	LINEAGE1("QTextEditItemLua")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);

	QGraphicsProxyWidget* item() {return proxy;}
	QTextEdit* widget() {return textedit;}

	void setTransparentBackgound(float t);

	QLuaHilighter* highlighter;
//signals:

//public slots:
//	void pressed();

private:
	QTextEdit* textedit;
	QGraphicsProxyWidget* proxy;

};

#endif // QTEXTEDITITEMLUA_H
