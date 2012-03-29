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
    virtual int luaInit(lua_State* ) {return 0;}
	virtual void push(lua_State* L) {luaT_push<Draw>(L, this);}
	
	Draw(int etype=0) : LuaBaseObject(etype) {}
    virtual void draw(Sphere& ) {}
    virtual void draw(Camera& ) {}
    virtual void draw(Light& ) {}
    virtual void draw(Tube& ) {}
    virtual void draw(Group& ) {}
    virtual void draw(Transformation& ) {}
	//virtual void draw(VolumeLua& volumelua) {}
	virtual void reset() {}
	
};

#endif

