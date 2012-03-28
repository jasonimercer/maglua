#ifndef TRANSFORMATION_Translate_H
#define TRANSFORMATION_Translate_H

#include <iostream>
using namespace std;

extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}
#include "Transformation.h"

class Translate : public Transformation
{
public:
	Translate() : Transformation(hash32("Translate")) {}
	~Translate() {};

	LINEAGE2("Translate", "Transformation")
	static const luaL_Reg* luaMethods(){return Transformation::luaMethods();}
	virtual int luaInit(lua_State* L)  {return Transformation::luaInit(L);}
	virtual void push(lua_State* L)    {luaT_push<Translate>(L, this);}
};

#endif
