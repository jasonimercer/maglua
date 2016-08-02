#ifndef QSLIDERITEMLUA_H
#define QSLIDERITEMLUA_H


#include <QSlider>
#include "QItemLua.h"

class QSliderItemLua : public QItemLua
{
public:
	QSliderItemLua();
	~QSliderItemLua();

	LINEAGE2("QSliderItemLua", "QItemLua")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);

	virtual void setTransparent(float t=1.0);
	QSlider* widget() {return slider;}

	SignalSink* changeFunc;
private:
	QSlider* slider;

};


#endif // QSLIDERITEMLUA_H
