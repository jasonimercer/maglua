#ifndef TUBE_H
#define TUBE_H

#include "Vector.h"
#include "Color.h"
#include "Ray.h"
#include "Volume.h"
#include "Color.h"

#include <string>
using namespace std;

extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}


class Tube : public Volume
{
public:
	Tube();
	Tube(const Tube& other);
	
	LINEAGE2("Tube", "Volume")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	
	void setPos(int i, const Vector& p);
	Vector pos(int i) const;
	
	void setRadius(int i, double r);
	double radius(int i) const;
	
	bool rayIntersect(const Ray& ray, double& t);
	
	double volume();
	
	virtual bool contains(const Vector& v, double expand=0.0);

protected:
	virtual void updateBoundingBox();

private:
	Vector _pos[2];
	double _radius[2];
	Vector _directions[3];
};



#endif // TUBE_H
