#include "Camera.h"
#include "Matrix.h"
#include <math.h>
#include <string.h>

using namespace std;

Camera::Camera()
	: LuaBaseObject(hash32("Camera"))
{
	at = up = right = pos = forward = 0;
    reset();
	ratio = 1.0;
	perspective = true;
}


int Camera::luaInit(lua_State* L)
{
	reset();
	return 0;
}
void Camera::push(lua_State* L)
{
	luaT_push<Camera>(L, this);
}
	
void Camera::reset()
{
	luaT_dec<Vector>(at);
	luaT_dec<Vector>(up);
	luaT_dec<Vector>(right);
	luaT_dec<Vector>(pos);
	luaT_dec<Vector>(forward);
	
	at = luaT_inc<Vector>(new Vector(0, 0, 0));
	up = luaT_inc<Vector>(new Vector(0, 0, 1));
	right  = luaT_inc<Vector>(new Vector(1, 0, 0));
	pos = luaT_inc<Vector>(new Vector(0, -10, 0));
	forward =  luaT_inc<Vector>(new Vector( (*at - *pos).normalized() ));
	FOV = 45; //degrees
}

Camera::~Camera()
{
	luaT_dec<Vector>(at);
	luaT_dec<Vector>(up);
	luaT_dec<Vector>(right);
	luaT_dec<Vector>(pos);
	luaT_dec<Vector>(forward);
}


double Camera::dist()
{
    return (*at-*pos).length();
}


void Camera::setDist(double d)
{
	Vector at2pos = *pos - *at;
	*pos = *at + d * at2pos.normalized();
}


void Camera::translate(const Vector& d)
{
    *pos += d;
	*at += d;
}

void Camera::translateUVW(const Vector& d)
{
    Vector r =   *right * d.x();
    Vector u =      *up * d.y();
    Vector f = *forward * d.z();

	translate(r + u + f);
}

void Camera::rotate(double theta, double phi)
{
	*pos -= at;

	right->rotateAbout(*up, theta);
	pos->rotateAbout(*up, theta);
	//up->rotateAbout(up, theta);
	forward->rotateAbout(*up, theta);

	//right->rotateAbout(right, phi);
	pos->rotateAbout(*right, phi);
	up->rotateAbout(*right, phi);
	forward->rotateAbout(*right, phi);

	*pos += at;
}

void Camera::rotateAbout(double theta, const Vector& vec)
{
	*pos -= *at;

	right->rotateAbout(vec, theta);
	pos->rotateAbout(vec, theta);
	up->rotateAbout(vec, theta);
	forward->rotateAbout(vec, theta);

	*pos += *at;
}


void Camera::zoom(double val)
{
	*pos += *forward * val;
}



void Camera::roll(double val)
{
    double theta = val * 0.00001;

    right->rotateAbout(*forward, theta);
    up->rotateAbout(*forward, theta);
}






#define getsetvecmacro(name)                \
                                            \
static int l_camera_get##name (lua_State* L)\
{                                           \
	LUA_PREAMBLE(Camera, c, 1);             \
	luaT_push<Vector>(L, c->name); \
	return 1;                               \
}                                           \
                                            \
static int l_camera_set##name (lua_State* L)\
{                                           \
	LUA_PREAMBLE(Camera, c, 1);             \
    Vector v;                               \
    lua_makevector(L, 2, v);                \
    c->name->set(v.vec());                  \
	return 0;                               \
}

getsetvecmacro(right)
getsetvecmacro(up)
getsetvecmacro(at)
getsetvecmacro(forward)
getsetvecmacro(pos)

#define getsetreg(name, cname) {#name, l_camera_get##name}, {"set"#cname, l_camera_set##name}


static int l_camera_reset(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	c->reset();
	return 0;
}

static int l_camera_rotate(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	
	double t = lua_tonumber(L, 2);
	double p = lua_tonumber(L, 3);
	c->rotate(t, p);
	return 0;
}

static int l_camera_rotateabout(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	
	double t = lua_tonumber(L, 2);
	
	Vector v;
	lua_makevector(L, 3, v);
	
	c->rotateAbout(t, v);
	return 0;
}

static int l_camera_roll(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	
	c->roll(lua_tonumber(L, 2));
	return 0;
}

static int l_camera_zoom(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	
	c->zoom(lua_tonumber(L, 2));
	return 0;
}

static int l_camera_translate(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	
	Vector v;
	lua_makevector(L, 2, v);
	c->translate(v);
	return 0;
}

static int l_camera_translateuvw(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	
	Vector v;
	lua_makevector(L, 2, v);
	c->translateUVW(v);
	return 0;
}

static int l_camera_dist(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	
	lua_pushnumber(L, c->dist());
	return 1;
}


static int l_camera_setdist(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	
	c->setDist(lua_tonumber(L, 2));
	
	return 0;
}

static int l_camera_at(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	
	luaT_push<Vector>(L, new Vector(c->at));
	return 1;
}


static int l_camera_getperspective(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	
	lua_pushboolean(L, c->perspective);
	return 1;
}
static int l_camera_setperspective(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	
	c->perspective = lua_toboolean(L, 2);
	return 0;
}

static int l_camera_makepose(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	
	Vector pos, at, up;
	lua_makevector(L, 2, pos);
	lua_makevector(L, 3, at);
	lua_makevector(L, 4, up);
	
	Vector forward = (at - pos).normalized();	
	Vector right = forward.cross(up);
	up = right.cross(forward);
	
	Matrix* m = new Matrix();
	m->makeIdentity();
	for(int i=0; i<3; i++)
	{
		m->setComponent(0, i, pos.component(i));
		m->setComponent(1, i, at.component(i));
		m->setComponent(2, i, up.component(i));
	}
	
	luaT_push<Matrix>(L, m);
	return 1;
}

static int l_camera_setpose(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	
	Vector pos, at, up;
	if(luaT_is<Matrix>(L, 2))
	{
		Matrix* m = luaT_to<Matrix>(L, 2);
		for(int i=0; i<3; i++)
		{
			pos.setComponent(i, m->component(0,i));
			 at.setComponent(i, m->component(1,i));
			 up.setComponent(i, m->component(2,i));
		}
	}
	else
	{
		lua_makevector(L, 2, pos);
		lua_makevector(L, 3, at);
		lua_makevector(L, 4, up);
	}
	
	*c->pos = pos;
	*c->at  =  at;
	
	*c->forward = (at - pos).normalized();
	*c->right = c->forward->cross(up);
	*c->up = c->right->cross(c->forward);
	
	return 0;
}
static int l_camera_getpose(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	Matrix* m = new Matrix();
	m->makeIdentity();
	for(int i=0; i<3; i++)
	{
		m->setComponent(0, i, c->pos->component(i));
		m->setComponent(1, i, c->at->component(i));
		m->setComponent(2, i, c->up->component(i));
	}
	luaT_push<Matrix>(L, m);
	return 1;
}

static int l_camera_getratio(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	lua_pushnumber(L, c->ratio);
	return 1;
}

static int l_camera_setratio(lua_State* L)
{
	LUA_PREAMBLE(Camera, c, 1);
	c->ratio = lua_tonumber(L, 2);
	return 0;
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Camera::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LuaBaseObject::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"reset",       l_camera_reset},
		{"ratio",       l_camera_getratio},
		{"setRatio",       l_camera_setratio},
		{"at",          l_camera_at},
		{"rotate",      l_camera_rotate},
		{"rotateAbout", l_camera_rotateabout},
		{"roll",        l_camera_roll},
		{"zoom",        l_camera_zoom},
		{"translate",   l_camera_translate},
		{"translateUVW",l_camera_translateuvw},
		{"dist",        l_camera_dist},
		{"setDist",     l_camera_setdist},
		{"perspective",     l_camera_getperspective},
		{"setPerspective",  l_camera_setperspective},
		{"pose",        l_camera_getpose},
		{"setPose",     l_camera_setpose},
		{"makePose",    l_camera_makepose},
		getsetreg(right, Right),
		  getsetreg(up, Up),
		  getsetreg(at, At),
		  getsetreg(forward, Forward),
		  getsetreg(pos, Pos),
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}



