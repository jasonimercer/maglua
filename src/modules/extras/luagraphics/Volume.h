#ifndef VOLUMES_H
#define VOLUMES_H

#include "Ray.h"
#include "Color.h"
#include "luabaseobject.h"
class AABB;

#include "libLuaGraphics.h"
class LUAGRAPHICS_API Volume : public LuaBaseObject
{
public:
	Volume(int etype=0);
	~Volume();
	
	LINEAGE1("Volume")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	
	virtual double volume() {return 0;}
    virtual bool rayIntersect(const Ray& /*ray*/, double& /*t*/) {return false;}
    virtual bool contains(const Vector& /*v*/, double /*expand=0.0*/) {return false;}
    virtual bool excludes(const Vector& /*v*/, double /*expand=0.0*/) {return true;}

	virtual AABB* getBB();
	virtual void setColor(Color* c);
	
	bool selected;
	Color* color;

protected:
	virtual void updateBoundingBox() {};
	AABB* bb;
};

#endif
