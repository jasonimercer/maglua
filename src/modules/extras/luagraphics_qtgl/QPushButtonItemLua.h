#ifndef QPUSHBUTTONITEMLUA_H
#define QPUSHBUTTONITEMLUA_H

#include <QPushButton>
#include "QItemLua.h"


class QPushButtonItemLua : public QItemLua
{
public:
	QPushButtonItemLua();
	~QPushButtonItemLua();

	LINEAGE2("QPushButtonItemLua", "QItemLua")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);

	virtual void setTransparent(float t=1.0);

	QPushButton* widget() {return pushbutton;}

	QPushButton* pushbutton;
	SignalSink* pressFunc;
};

#endif // QPUSHBUTTONITEMLUA_H
