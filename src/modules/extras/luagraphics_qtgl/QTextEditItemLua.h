#ifndef QTEXTEDITITEMLUA_H
#define QTEXTEDITITEMLUA_H

#include <QTextEdit>
#include "QLuaHilighter.h"
#include <QGraphicsProxyWidget>
#include "QItemLua.h"

#include "luabaseobject.h"
extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}

class QTextEditItemLua : public QItemLua
{
public:
	QTextEditItemLua();
	~QTextEditItemLua();

	LINEAGE2("QTextEditItemLua", "QItemLua")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);

	QTextEdit* widget() {return textedit;}

	void setTransparentBackgound(float t);

	QLuaHilighter* highlighter;

private:
	QTextEdit* textedit;

};

#endif // QTEXTEDITITEMLUA_H
