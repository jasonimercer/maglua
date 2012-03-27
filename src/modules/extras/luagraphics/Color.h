#ifndef COLOR_H
#define COLOR_H
#include "Vector.h"
#include "luabaseobject.h"

class Color : public LuaBaseObject
{
public:
	Color(double r=0.0, double g=0.0, double b=0.0, double a=1.0);
	void set(Color* c);
	double rgba[4];	
	
	LINEAGE1("Color")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	
	double r() const {return rgba[0];};
	double g() const {return rgba[1];};
	double b() const {return rgba[2];};
	double a() const {return rgba[3];};
	double t() const {return 1.0-a();};
	
	void  setComponent(int i, double v);
	double component(int i) const;
	
	Color& operator =(const Color& rhs);
	Color& operator =(const Vector& rhs);
	
	Vector toVector() const {return Vector(rgba);} 
	
};

#endif

