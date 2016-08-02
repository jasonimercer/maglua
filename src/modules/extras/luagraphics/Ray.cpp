#include "Ray.h"

Ray::Ray()
	: LuaBaseObject(hash32("Ray"))
{
	origin = luaT_inc<Vector>(new Vector(0,0,0));
	direction = luaT_inc<Vector>(new Vector(1,0,0));
}

Ray::Ray(const Vector& o, const Vector& d)
	: LuaBaseObject(hash32("Ray"))
{
	*origin = o;
	*direction = d;
}

Ray::Ray(const Ray& r)
	: LuaBaseObject(hash32("Ray"))
{
	*this = r;
}

Ray::~Ray()
{
	luaT_dec<Vector>(origin);
	luaT_dec<Vector>(direction);
}

void Ray::push(lua_State* L)
{
	luaT_push<Ray>(L, this);
}

int Ray::luaInit(lua_State* L)
{
	if(luaT_is<Ray>(L, 1))
	{
		(*this) = *(luaT_to<Ray>(L, 1));
	}
	if(luaT_is<Vector>(L, 1))
	{
		*origin = *(luaT_to<Vector>(L, 1));
	}
	if(luaT_is<Vector>(L, 2))
	{
		*direction = *(luaT_to<Vector>(L, 2));
	}
	return 0;
}


Vector Ray::operator() (double t) const
{
	return *origin + t * (*direction);
}

Ray& Ray::operator=(const Ray& rhs)
{
	*origin = *rhs.origin;
	*direction = *rhs.direction;

	return *this;
}

bool Ray::operator==(const Ray &other) const
{
	return *origin == *other.origin && *direction == *other.direction;
}









int lua_makeray(lua_State* L, int idx, Ray& ray)
{
	if(luaT_is<Ray>(L, idx))
	{
		Ray* b = luaT_to<Ray>(L, idx);
		if(b)
		{
			ray = *b;
		}
		return 1;
	}
	
	int r = 0;
	
	r += lua_makevector(L, idx, *ray.origin);
	r += lua_makevector(L, idx+r, *ray.direction);
	
	return r;
}


static int l_ray_tostring(lua_State* L)
{
	LUA_PREAMBLE(Ray, r, 1);
	
	const Vector& o = *r->origin;
	const Vector& d = *r->direction;
	
	lua_pushfstring(L, "(%f, %f, %f) (%f, %f, %f) ", o.x(), o.y(), o.z(), d.x(), d.y(), d.z());
	return 1;
}

static int l_ray_eq(lua_State* L)
{
	LUA_PREAMBLE(Ray, a, 1);
	LUA_PREAMBLE(Ray, b, 2);
	
	lua_pushboolean(L, (*a) == (*b));
	return 1;
}

static int l_ray_setorigin(lua_State* L)
{
	LUA_PREAMBLE(Ray, r, 1);
	
	lua_makevector(L, 2, *r->origin);
	return 0;
}

static int l_ray_getorigin(lua_State* L)
{
	LUA_PREAMBLE(Ray, r, 1);
	luaT_push<Vector>(L, new Vector(r->origin));
	return 1;
}


static int l_ray_setdirection(lua_State* L)
{
	LUA_PREAMBLE(Ray, r, 1);
	
	lua_makevector(L, 2, r->direction);
	return 0;
}

static int l_ray_getdirection(lua_State* L)
{
	LUA_PREAMBLE(Ray, r, 1);
	luaT_push<Vector>(L, new Vector(r->direction));
	return 1;
}



static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Ray::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LuaBaseObject::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"__tostring",   l_ray_tostring},
		{"__eq",         l_ray_eq},
		{"setOrigin",    l_ray_setorigin},
		{"origin",       l_ray_getorigin},
		{"setDirection", l_ray_setdirection},
		{"direction",    l_ray_getdirection},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}


