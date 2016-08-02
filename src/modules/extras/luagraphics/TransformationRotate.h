#ifndef TRANSFORMATION_Rotate_H
#define TRANSFORMATION_Rotate_H

#include <iostream>
using namespace std;

extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}
#include "Transformation.h"

#include "libLuaGraphics.h"
class LUAGRAPHICS_API Rotate : public Transformation
{
public:
	Rotate() : Transformation(hash32("Rotate")) {}
	~Rotate() {};

	LINEAGE2("Rotate", "Transformation")
	static const luaL_Reg* luaMethods(){return Transformation::luaMethods();}
	virtual int luaInit(lua_State* L)  {return Transformation::luaInit(L);}
	virtual void push(lua_State* L)    {luaT_push<Rotate>(L, this);}
};

#endif
