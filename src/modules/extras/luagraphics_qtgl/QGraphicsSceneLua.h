#ifndef QGRAPHICSSCENELUA_H
#define QGRAPHICSSCENELUA_H

#include "luabaseobject.h"
#include <QGraphicsScene>

class QGraphicsSceneLua : public LuaBaseObject
{
public:
	QGraphicsSceneLua(QGraphicsScene* scene=0);
	~QGraphicsSceneLua();

	LINEAGE1("QGraphicsSceneLua");
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);

	QGraphicsScene* scene;
};

#endif // QGRAPHICSSCENELUA_H
