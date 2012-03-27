#include "AABB.h"
#include "Plane.h"

AABB::AABB()
	: LuaBaseObject(hash32("AABB"))
{
	min = luaT_inc<Vector>(new Vector);
	max = luaT_inc<Vector>(new Vector);
	reset();
}

AABB::~AABB()
{
	luaT_dec<Vector>(min);
	luaT_dec<Vector>(max);
}

AABB* AABB::getBB()
{
	return this;
}

	
int AABB::luaInit(lua_State* L)
{
	reset();
	Vector* a=0;
	Vector* b=0;
	
	if(luaT_is<Vector>(L, 1))
		a = luaT_to<Vector>(L, 1);
	if(luaT_is<Vector>(L, 2))
		b = luaT_to<Vector>(L, 2);
	
	if(a && b)
	{
		include(*a);
		include(*b);
	}
	
	return 0;
}

void AABB::push(lua_State* L)
{
	luaT_push<AABB>(L, this);
}



double AABB::volume()
{
	if(!pointAdded)
		return 0;
	
	Vector a = (*max) - (*min);
	return a.x() * a.y() * a.z();
}

	
void AABB::reset()
{
	pointAdded = false;
}
	
void AABB::include(const Vector& v)
{
	if(!pointAdded)
	{
		*max = v;
		*min = v;
		pointAdded = true;
		return;
	}
	
	(*min) = Vector::min(*min, v);
	(*max) = Vector::max(*max, v);
}

void AABB::include(const AABB& bb)
{
	include(*(bb.max));
	include(*(bb.min));
}


bool AABB::contains(const Vector& v, double expand)
{
	if(!pointAdded)
		return false;
	
	for(int i=0; i<3; i++)
	{
		if(v.component(i) < (min->component(i) + expand))
			return false;
		if(v.component(i) > (max->component(i) - expand))
			return false;
	}
	return true;
}

static bool between(double m, double M, double x, double expand=0)
{
	return (m+expand) <= x && x <= (M-expand);
}

bool AABB::excludes(const AABB& bb, double expand)
{
	double isect = true;
	
	for(int i=0; i<3; i++)
	{
		const double a1 = min->component(i);
		const double b1 = max->component(i);
		
		const double a2 = bb.min->component(i);
		const double b2 = bb.max->component(i);
		
		const double v1 = between(a1, b1, a2, expand);
		const double v2 = between(a1, b1, b2, expand);
		const double v3 = between(a2, b2, a1, expand);
		const double v4 = between(a2, b2, b1, expand);
		
		if(!(v1 || v2 || v3 || v4))
			return true;
	}
	return false;
}


bool AABB::excludes(const Vector& v, double expand)
{
	if(!pointAdded)
		return true;

// 	bool res[6];
	
	for(int i=0; i<3; i++)
	{
		const double d = v.component(i);
		
		if(d > (max->component(i) + expand))
			return true;
		if(d < (min->component(i) - expand))
			return true;
		
// 		res[i*2+0] = d > max.component(i) + expand;
// 		res[i*2+1] = d < min.component(i) - expand;
/*		
		res[i] = 
		if( (d < max.component(i) + expand) && (d > min.component(i) - expand))
			return false;
		
		if(d > (min.component(i) - expand))
			return false;
		if(d < (max.component(i) + expand))
			return false;*/
	}
	return false;
}



bool AABB::rayIntersect(const Ray& ray, double& T)
{
	if(!pointAdded)
		return false;
	
	// Ray = origin + t direction
	// need to test the 6 faces for an intersection
	
	T = 1E10;
	
	double t[6];
	bool  b[6];
	
	b[0] = Plane(Vector(0,0,1), *max).rayIntersect(ray, t[0]);
	b[1] = Plane(Vector(0,0,1), *min).rayIntersect(ray, t[1]);
	b[2] = Plane(Vector(0,1,0), *max).rayIntersect(ray, t[2]);
	b[3] = Plane(Vector(0,1,0), *min).rayIntersect(ray, t[3]);
	b[4] = Plane(Vector(1,0,0), *max).rayIntersect(ray, t[4]);
	b[5] = Plane(Vector(1,0,0), *min).rayIntersect(ray, t[5]);
	
	bool r = false;
	for(int i=0; i<6; i++)
	{
		if(b[i])
		{
			if(contains(ray(t[i]), 1.01))
			{
				r = true;
				if(t[i] < T)
					T = t[i];
			}
		}
	}
	
	return r;
}





static int l_include(lua_State* L)
{
	LUA_PREAMBLE(AABB, aabb, 1);
	
	if(luaT_is<Vector>(L, 2))
	{
		aabb->include(* luaT_to<Vector>(L, 2));
	}
	if(luaT_is<Vector>(L, 2))
	{
		aabb->include(* luaT_to<AABB>(L, 2));
	}
	
	return 0;
}
static int l_volume(lua_State* L)
{
	LUA_PREAMBLE(AABB, v, 1);
	lua_pushnumber(L, v->volume());
	return 1;
}

static int l_rayisect(lua_State* L)
{
	LUA_PREAMBLE(AABB, v, 1);
	LUA_PREAMBLE(Ray, r, 1);
	double t;
	bool b = v->rayIntersect(*r, t);
	lua_pushboolean(L, b);
	lua_pushnumber(L, t);
	return 2;	
}

static int l_contains(lua_State* L)
{
	LUA_PREAMBLE(AABB, v, 1);
	LUA_PREAMBLE(Vector, vec, 2);
	double expand = 0;
	if(lua_isnumber(L, 3))
		expand = lua_tonumber(L, 3);
	
	lua_pushboolean(L, v->contains(*vec, expand));
	return 1;
}

static int l_excludes(lua_State* L)
{
	LUA_PREAMBLE(AABB, v, 1);
	LUA_PREAMBLE(Vector, vec, 2);
	double expand = 0;
	if(lua_isnumber(L, 3))
		expand = lua_tonumber(L, 3);
	
	lua_pushboolean(L, v->excludes(*vec, expand));
	return 1;
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* AABB::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LuaBaseObject::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"volume",       l_volume},
		{"rayIntersect", l_rayisect},
		{"contains",     l_contains},
		{"excludes",     l_excludes},
		{"include",      l_include},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}

