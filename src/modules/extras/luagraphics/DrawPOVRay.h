#ifndef DRAWPOVRAY_H
#define DRAWPOVRAY_H

#include <iostream>
#include <fstream>
using namespace std;

#include "Draw.h"
#include "Tube.h"
#include "Color.h"
#include "Sphere.h"
#include "VolumeLua.h"

extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}

class DrawPOVRay : public Draw
{
public:
	DrawPOVRay();
// 	DrawPOVRay(ostream& outstream);
// 	DrawPOVRay(const char* filename);
	~DrawPOVRay();

	LINEAGE2("DrawPOVRay", "Draw")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	
	void draw(Sphere& s);
	void draw(Camera& camera);
	void draw(Tube& tube);
	void draw(Light& light);
	//void draw(VolumeLua& volumelua);
	void draw(Group& group);
	void draw(Transformation& t);

	void setFileName(const char* filename);
private:
	void init();
	std::ofstream out;
};


#endif
