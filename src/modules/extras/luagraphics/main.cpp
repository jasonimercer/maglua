extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}

#include <iostream>
using namespace std;
// #include "libCrystal.h"

#include "libLuaGraphics.h"

int main(int argc, char** argv)
{
	lua_State* L = lua_open();
	luaL_openlibs(L);
	
// 	lua_registerlibcrystal(L);
	
	lua_RegisterGraphics(L);
	
	if(luaL_dofile(L, "test.lua"))
		cerr << lua_tostring(L, -1) << endl;
	
// 	VolumeLua VL;
// 	
// 	vector<double> p;
// 	p.push_back(0.0);
// 	p.push_back(0.0);
// 	p.push_back(0.0);
// 	p.push_back(1.0);
// 	
// 	VL.getFaces(L, p);
// 
// 	cout << VL.volume() << endl;
// 	
// 	lua_close(L);
// 	
// 	Atom a;
// 	Camera c;
// 	
// 	DrawPOVRay dp(cout);
// 	dp.init(c);
// 	
// 	a.draw(dp);
	
	return 0;
}
