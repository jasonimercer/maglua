#ifndef DRAW_H
#define DRAW_H

// this class is the abstract base class for drawing functions
// the POVRay and OpenGL are implementations of it.

#include "Group.h"
#include "Sphere.h"
#include "Color.h"
#include "Camera.h"
#include "Light.h"
#include "Tube.h"
//#include "VolumeLua.h"
#include "luabaseobject.h"
#include "Transformation.h"
class Draw : public LuaBaseObject
{
public:
	LINEAGE1("Draw")
	
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L) {return 0;}
	virtual void push(lua_State* L) {luaT_push<Draw>(L, this);}
	
	Draw(int etype=0) : LuaBaseObject(etype) {}
	virtual void draw(Sphere& s) {}
	virtual void draw(Camera& camera) {}
	virtual void draw(Light& light) {}
	virtual void draw(Tube& tube) {}
	virtual void draw(Group& group) {}
	virtual void draw(Transformation& t) {}
	//virtual void draw(VolumeLua& volumelua) {}
	virtual void reset() {}
	
};

#endif

