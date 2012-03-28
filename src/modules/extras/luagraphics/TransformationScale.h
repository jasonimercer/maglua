#ifndef TRANSFORMATION_Scale_H
#define TRANSFORMATION_Scale_H

#include <iostream>
using namespace std;

extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}
#include "Transformation.h"

class Scale : public Transformation
{
public:
	Scale() : Transformation(hash32("Scale")) {type = scale;};
	~Scale() {};

	LINEAGE2("Scale", "Transformation")
	static const luaL_Reg* luaMethods(){return Transformation::luaMethods();}
	virtual int luaInit(lua_State* L)  {return Transformation::luaInit(L);}
	virtual void push(lua_State* L)    {luaT_push<Scale>(L, this);}
};

#endif
