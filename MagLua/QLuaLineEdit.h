#ifndef QLUALINEEDIT_H
#define QLUALINEEDIT_H

#include <QLineEdit>
#include "QLuaWidget.h"

class QLuaLineEdit : public QLuaWidget
{
public:
	explicit QLuaLineEdit(lua_State* L, QLineEdit* w) : QLuaWidget(L, w) {}

};



int lua_islualineedit(lua_State* L, int idx);
QLuaLineEdit* lua_tolualineedit(lua_State* L, int idx);
void lua_pushlualineedit(lua_State* L, QLuaLineEdit* s);
void lua_registerlineedit(lua_State* L);

#endif // QLUALINEEDIT_H
