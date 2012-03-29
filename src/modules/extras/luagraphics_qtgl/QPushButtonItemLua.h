#ifndef QPUSHBUTTONITEMLUA_H
#define QPUSHBUTTONITEMLUA_H

#include <QPushButton>
#include "luabaseobject.h"
#include "SignalSink.h"

class QPushButtonItemLua : public LuaBaseObject
{
public:
	QPushButtonItemLua();
	~QPushButtonItemLua();

	LINEAGE1("QPushButtonItemLua")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);

	QGraphicsProxyWidget* item() {return proxy;}
	QPushButton* widget() {return pushbutton;}

private:
	QPushButton* pushbutton;
	QGraphicsProxyWidget* proxy;

	SignalSink* pressFunc;

};

#endif // QPUSHBUTTONITEMLUA_H
