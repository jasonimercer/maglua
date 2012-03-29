#ifndef QGRAPHICSITEMLUA_H
#define QGRAPHICSITEMLUA_H

#include "luabaseobject.h"
#include <QGraphicsItem>

class QGraphicsItemLua : public LuaBaseObject
{
public:
	QGraphicsItemLua(QGraphicsItem* item=0);
	~QGraphicsItemLua();

	LINEAGE1("QGraphicsItemLua");
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);

	QGraphicsItem* item;
};

#endif // QGRAPHICSITEMLUA_H
