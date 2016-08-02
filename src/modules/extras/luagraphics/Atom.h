class Atom;

#ifndef ATOM_H
#define ATOM_H

#include "Draw.h"
#include "Sphere.h"

#include <string>
using namespace std;

extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}

// class Draw;
class Atom : public Sphere
{
public:
    Atom(lua_State* _L);

	Atom(const Atom& other);
	Atom(lua_State* _L, const Sphere& shape, string name="", string type="");
	~Atom();
	
// 	void updateBoundingBox();
	

// 	void draw(Draw& drawfuncs) const;
// 	void draw(const DrawOptions* drawopts, const CameraGL* cam);
// 	bool rayIntersect(const DrawOptions* drawopts, Ray* ray, qreal* t=0);
	


// 	int selected;

	
	string name;
	string type;
	int refcount; //for lua
	
	
	double occupancy;
	double vdwRadius;
	
	int dataRef;
	lua_State* L;
};


Atom* lua_toatom(lua_State* L, int idx);
int  lua_isatom(lua_State* L, int idx);
void lua_pushatom(lua_State* L, Atom* a);
void lua_registeratom(lua_State* L);


#endif // ATOM_H
