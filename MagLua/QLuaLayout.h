#ifndef QLUALAYOUT_H
#define QLUALAYOUT_H

#include <QGridLayout>
#include "QLuaWidget.h"

class QLuaLayout : public QLuaWidget
{
public:
	QLuaLayout(lua_State* L, QLayout* _layout) : QLuaWidget(L, 0), layout(_layout) {}
	~QLuaLayout() {if(layout) delete layout;}

	QLayout* layout;
};

int lua_islayout(lua_State* L, int idx);
QLuaLayout* lua_tolayout(lua_State* L, int idx);
void lua_pushlualayout(lua_State* L, QLuaLayout* s);
void lua_registerlayout(lua_State* L);


#endif // QLUALAYOUT_H
