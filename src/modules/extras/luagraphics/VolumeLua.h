class VolumeLua;

#ifndef VOLUMELUA_H
#define VOLUMELUA_H

#include <vector>
#include <string>
using namespace std;

extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}

#include "Sphere.h"
#include "Vector.h"
#include "Tetrahedron.h"
#include "Volume.h"
#include "Color.h"
#include "Group.h"

class VolumeLua : public Volume
{
public:
	VolumeLua();
	~VolumeLua();

	void setFunction(lua_State* L, int funcref);
	
	bool getFaces(lua_State* L, int index);
	bool sphereIsect(const Sphere& s, int* t = 0) const;
	bool sphereEnclosed(const Sphere& s) const;

	void makeFace(const Vector& center, const vector<Vector>& face);

	bool rayIntersect(const Ray& ray, double& t);

	double volume();

	vector<Tetrahedron> tetrahedron;
	vector<string> pnames;

	Vector center;
	
	int refcount; //for lua
	
	Color color;

	virtual bool contains(const Vector& v, double expand=0.0);
	virtual bool excludes(const Vector& v, double expand=0.0);

// 	virtual bool contains(const Vector& v, double expand=0.0);
	bool excludes(Group* g);

protected:
	bool excludes(GroupNode* gn);

	virtual void updateBoundingBox();
	
	int funcref;
	lua_State* L;
};

VolumeLua* lua_tovolumelua(lua_State* L, int idx);
int  lua_isvolumelua(lua_State* L, int idx);
void lua_pushvolumelua(lua_State* L, VolumeLua* v);
void lua_registervolumelua(lua_State* L);


#endif // VOLUMELUA_H
