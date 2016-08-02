#ifndef QTEXTEDITITEMLUA_H
#define QTEXTEDITITEMLUA_H

#include <QTextEdit>
#include "QLuaHilighter.h"
#include "QItemLua.h"

class QTextEditItemLua : public QItemLua
{
public:
	QTextEditItemLua();
	~QTextEditItemLua();

	LINEAGE2("QTextEditItemLua", "QItemLua")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);

	virtual void setTransparent(float t=1.0);

	QTextEdit* widget() {return textedit;}

	QLuaHilighter* highlighter;

private:
	QTextEdit* textedit;

};

#endif // QTEXTEDITITEMLUA_H
