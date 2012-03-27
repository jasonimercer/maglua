#ifndef LIGHT_H
#define LIGHT_H

// #include "Draw.h"
#include "Sphere.h"
#include "Color.h"

#include <string>
using namespace std;

extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}

class Light : public Sphere
{
public:
	Light();

	LINEAGE3("Light", "Sphere", "Volume")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	
	Color* diffuse_color;
	Color* specular_color;
};



#endif // LIGHT_H
