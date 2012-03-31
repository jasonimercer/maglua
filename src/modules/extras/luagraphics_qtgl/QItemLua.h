#ifndef QITEMLUA_H
#define QITEMLUA_H

#include "luabaseobject.h"
#include <QGraphicsProxyWidget>
#include <QGraphicsScene>
#include "SignalSink.h"
#include "QGraphicsSceneLua.h"

class QItemLua : public LuaBaseObject
{
public:
	QItemLua(int etype = 0);
	~QItemLua();

	LINEAGE1("QItemLua")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);

	QGraphicsProxyWidget* item() {return proxy;}

	void setGeometry(int x, int y, int w, int h);

	QRect geometry();

protected:
	QGraphicsProxyWidget* proxy;
	QGraphicsScene* scene;

private:
	QGraphicsSceneLua* scene_lua;

};

#endif // QITEMLUA_H
