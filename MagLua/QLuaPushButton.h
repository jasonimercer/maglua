#ifndef QLUAPUSHBUTTON_H
#define QLUAPUSHBUTTON_H

#include <QPushButton>
#include "QLuaWidget.h"

class QLuaPushButton : public QLuaWidget
{
Q_OBJECT
public:
	QLuaPushButton(lua_State* L, QPushButton* w);
	~QLuaPushButton();

	void setPressedFunction(int ref);
	int funcref;

public slots:
	void pressed();
};


int lua_ispushbutton(lua_State* L, int idx);
QLuaPushButton* lua_topushbutton(lua_State* L, int idx);
void lua_pushluapushbutton(lua_State* L, QLuaPushButton* s);
void lua_registerpushbutton(lua_State* L);

#endif // QLUAPUSHBUTTON_H
