#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include <iostream>
using namespace std;

extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}
#include "luabaseobject.h"
#include "Volume.h"

#include <vector>
using namespace std;

class Transformation : public LuaBaseObject
{
public:
	Transformation();
	Transformation(int etype);
	~Transformation();

	LINEAGE1("Transformation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);

	void addVolume(Volume* v);
	void setNextTransformation(Transformation* nt);
	
	Transformation* nextTransform;
	vector<Volume*> volumes;
	
	enum TransformationType{none, rotate, scale, translate};
	TransformationType type;
	
	double values[3];
};

#endif
