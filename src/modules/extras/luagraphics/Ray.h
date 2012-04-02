#ifndef RAY_H
#define RAY_H

#include "Vector.h"
#include "luabaseobject.h"

#include "libLuaGraphics.h"
class LUAGRAPHICS_API Ray : public LuaBaseObject
{
public:
	Ray();
	Ray(const Vector& o, const Vector& d);
	Ray(const Ray& r);
	~Ray();
	
	LINEAGE1("Ray")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	
	Vector operator() (double t) const;
	Ray& operator =(const Ray& rhs);
	bool operator==(const Ray &other) const;
	
	Vector* origin;
	Vector* direction;
};

//int lua_makeray(lua_State* L, int idx, Ray& r);

#endif

