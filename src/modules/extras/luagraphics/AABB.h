#ifndef AABB_H
#define AABB_H

#include "Vector.h"
#include "Ray.h"
#include "luabaseobject.h"
#include "libLuaGraphics.h"

class LUAGRAPHICS_API AABB : public LuaBaseObject
{
public:
	AABB();
	~AABB();
	
	LINEAGE1("AABB")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	
	void reset();
	
	void include(const Vector& v);
	void include(const AABB& bb);
	

	virtual bool contains(const Vector& v, double expand=0.0);
	virtual bool excludes(const Vector& v, double expand=0.0);
	virtual bool excludes(const AABB& bb,  double expand=0.0);

	virtual double volume();
	virtual bool rayIntersect(const Ray& ray, double& t);
	
	virtual AABB* getBB();
	
// private:
	Vector* min;
	Vector* max;
	bool pointAdded;
};

#endif
