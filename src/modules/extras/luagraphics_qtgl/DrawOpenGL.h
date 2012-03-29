#ifndef DRAWOPENGL_H
#define DRAWOPENGL_H

#include "Draw.h"
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

    LINEAGE2("DrawOpenGL", "Draw")
    static const luaL_Reg* luaMethods();
    virtual int luaInit(lua_State* L);
    virtual void push(lua_State* L);

    void draw(Sphere& s);
    void draw(Camera& camera);
    void draw(Tube& tube);
    void draw(Light& light);
    //void draw(VolumeLua& volumelua);
	//void draw(Group& group);
    void draw(Transformation& t);

    virtual void reset();

private:
    void init();
    GLUquadric* q;
    int next_light;
};

#endif // DRAWOPENGL_H
