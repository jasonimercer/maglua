#ifndef CAMERABASE_H
#define CAMERABASE_H

#include "Vector.h"

extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}
#include "luabaseobject.h"

class Camera : public LuaBaseObject
{
public:
	Camera();
	~Camera();
	
	LINEAGE1("Camera")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	
	void reset();

	void rotate(double theta, double phi);
	void rotateAbout(double theta, const Vector& vec);

	void roll(double val);
	void zoom(double val);
	void setDist(double d);
	
	void translate(const Vector& d);
	void translateUVW(const Vector& d);

	double dist();

	double ratio;
	double FOV;
	
	Vector* right;
	Vector* up;
	Vector* at;
	Vector* forward;
	Vector* pos;
	
	bool perspective;
};


#endif

