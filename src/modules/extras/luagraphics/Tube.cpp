#include "Tube.h"

static int clamp2(int& i)
{
	if(i < 0) i = 0;
	if(i > 1) i = 1;
	return i;
}

Tube::Tube()
{
	setPos(0, Vector(0,0,0));
	setPos(1, Vector(0,0,1));
	setRadius(0, 0.5);
	setRadius(1, 0.5);
	refcount = 0;
}

int Tube::luaInit(lua_State* L)
{

	return 0;
}

void Tube::push(lua_State* L)
{
	luaT_push<Tube>(L, this);
}

Tube::Tube(const Tube& other)
{
	for(int i=0; i<2; i++)
	{
		setPos(i, other.pos(i));
		setRadius(i, other.radius(i));
	}
}

bool Tube::contains(const Vector& v, double expand)
{
	return false;
}

void Tube::updateBoundingBox()
{
}

	
void Tube::setPos(int i, const Vector& p)
{
	clamp2(i);
	_pos[i] = p;
	
	_directions[0] = _pos[1] - _pos[0];
	if(_directions[0].lengthSquared() == 0)
	{
		_directions[0] = Vector(0,0,1);
		_directions[1] = Vector(1,0,0);
		_directions[2] = Vector(0,1,0);
		return;
	}

	Vector a = _directions[0];
	a.randomize();
	
	while(a.dot(_directions[0]) == 0)
		a.randomize();
	
	_directions[1] = _directions[0].cross(a);
	_directions[2] = _directions[1].cross(_directions[0]);

	for(int i=0; i<3; i++)
		_directions[i].normalize();
}

Vector Tube::pos(int i) const
{
	clamp2(i);
	return _pos[i];
}

	
void Tube::setRadius(int i, double r)
{
	clamp2(i);
	_radius[i] = r;
}

double Tube::radius(int i) const
{
	clamp2(i);
	return _radius[i];
}


bool Tube::rayIntersect(const Ray& ray, double& t)
{
	return false;
}


double Tube::volume()
{
	return 0;
}










static int l_tube_eq(lua_State* L)
{
	LUA_PREAMBLE(Tube, a, 1);
	LUA_PREAMBLE(Tube, b, 2);

	bool p0 = a->pos(0) == b->pos(0);
	bool p1 = a->pos(1) == b->pos(1);
	
	bool r0 = a->radius(0) == b->radius(0);
	bool r1 = a->radius(1) == b->radius(1);
	
	lua_pushboolean(L,  p0 && p1 && r0 && r1);
	return 1;
}

static int l_tube_getpos(lua_State* L)
{
	LUA_PREAMBLE(Tube, a, 1);
	
	int idx = lua_tointeger(L, 2) - 1;
	clamp2(idx);

	luaT_push<Vector>(L, new Vector(a->pos(idx)));
	return 1;
}

static int l_tube_setpos(lua_State* L)
{
	LUA_PREAMBLE(Tube, a, 1);

	int idx = lua_tointeger(L, 2);
	
	Vector v;
	lua_makevector(L, 3, v);
	
	a->setPos(idx-1, v);
 	return 0;
}


static int l_tube_setradius(lua_State* L)
{
	LUA_PREAMBLE(Tube, a, 1);
	int idx = lua_tointeger(L, 2)-1;
	clamp2(idx);
	a->setRadius(idx, lua_tonumber(L, 3));
	return 0;
}
static int l_tube_getradius(lua_State* L)
{
	LUA_PREAMBLE(Tube, a, 1);
	int idx = lua_tointeger(L, 2)-1;
	clamp2(idx);

	lua_pushnumber(L, a->radius(idx));
	return 1;
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Tube::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, Volume::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"__eq",         l_tube_eq},
		{"setPosition",       l_tube_setpos},
		{"setRadius",    l_tube_setradius},
		{"position",          l_tube_getpos},
		{"radius",       l_tube_getradius},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}
