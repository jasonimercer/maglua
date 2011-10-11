#ifndef QLUASPLITTER_H
#define QLUASPLITTER_H

#include <QSplitter>
#include "QLuaWidget.h"

class QLuaSplitter : public QLuaWidget
{
public:
	QLuaSplitter(lua_State* L, QSplitter* s) : QLuaWidget(L, s) {}
};


int lua_issplitter(lua_State* L, int idx);
QLuaSplitter* lua_tosplitter(lua_State* L, int idx);
void lua_pushluasplitter(lua_State* L, QLuaSplitter* s);
void lua_registersplitter(lua_State* L);



#endif // QLUASPLITTER_H
