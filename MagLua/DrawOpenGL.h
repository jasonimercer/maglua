#ifndef DRAWOPENGL_H
#define DRAWOPENGL_H

#include <libLuaGraphics.h>
#include <QGLWidget>

extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}

class DrawOpenGL : public Draw
{
public:
	DrawOpenGL();
	~DrawOpenGL();

	void draw(const Sphere& s, const Color& c);
	void draw(Camera& camera);
	void draw(Light& light);
	void draw(Tube& tube);
	void draw(Group& group);
	void draw(VolumeLua& volumelua);

	void reset();

	int refcount; //for lua
private:
	void init();
	GLUquadric* q;
	int next_light;
};

int lua_isdrawopengl(lua_State* L, int idx);
DrawOpenGL* lua_todrawopengl(lua_State* L, int idx);
void lua_pushdrawopengl(lua_State* L, DrawOpenGL* d);
void lua_registerdrawopengl(lua_State* L);

#endif // DRAWOPENGL_H
