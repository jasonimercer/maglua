#include "Sphere.h"
#include <math.h>
#include "AABB.h"

Sphere::Sphere()
	: Volume(hash32("Sphere"))
{
	_radius = 1.0;
	_pos = luaT_inc<Vector>(new Vector(0,0,0));
	updateBoundingBox();
}
	
Sphere::Sphere(int etype)
	: Volume(etype)
{
	_radius = 1.0;
	_pos = luaT_inc<Vector>(new Vector(0,0,0));
	updateBoundingBox();
}
	
Sphere::~Sphere()
{
}

int Sphere::luaInit(lua_State* L)
{
	if(luaT_is<Sphere>(L, 1))
	{
		Sphere* b = luaT_to<Sphere>(L, 1);
		(*_pos) = *b->pos();
		_radius = b->radius();
		return 0;
	}
	{
		double v[4];
		bool ok = true;
		for(int i=0; i<4; i++)
		{
			if(lua_isnumber(L, i+1))
				v[i] = lua_tonumber(L, i+1);
			else
				ok = false;
		}
		
		if(ok)
		{
			Vector vec(v);
			setPos(&vec);
			setRadius(v[3]);
		}
		return 0;
	}
	if(luaT_is<Vector>(L, 1))
	{
		setPos(luaT_to<Vector>(L, 1));
		setRadius(lua_tonumber(L, 2));
	}
	return 0;
}

void Sphere::push(lua_State* L)
{
	luaT_push<Sphere>(L, this);
}
	
	
Sphere::Sphere(Sphere& other)
	: Volume(hash32("Sphere"))
{
	setPos(other.pos());
	setRadius(other.radius());
}

Sphere::Sphere(double x, double y, double z, double rad)
{
	_pos = luaT_inc<Vector>(new Vector(x,y,z));
	setRadius(rad);
}

void Sphere::setPos(Vector* p)
{
	*_pos = *p;
	updateBoundingBox();
}

Vector* Sphere::pos()
{
	return _pos;
}

void Sphere::setRadius(double r)
{
	_radius = r;
 	updateBoundingBox();
}

double Sphere::radius() const
{
	return _radius;
}


void Sphere::updateBoundingBox()
{
	const double r = radius();
	bb->reset();
	bb->include(*pos() - Vector(r, r, r)); 
	bb->include(*pos() + Vector(r, r, r)); 
}

bool Sphere::rayIntersect(const Ray& ray, double& t)
{
// 	if(!bb.rayIntersect(ray, t))
// 		return false;
	
	double a = ray.direction->lengthSquared();
	double b = 2.0 * ray.direction->dot(*ray.origin-pos());
	double c = (*ray.origin - *pos()).lengthSquared() - radius()*radius();
	
// 	Vector::dotProduct(pos(), pos()) +
// 			  Vector::dotProduct(ray.origin, ray.origin) -
// 		2.0 * Vector::dotProduct(pos(), ray.origin) -
// 		      radius()*radius();
// 
	double T0, T1;
	double bb4ac = b*b - 4.0 * a * c;

	if(bb4ac < 0)
		return false;

	bb4ac = sqrt(bb4ac);

	//two solutions to the quadratic equation
	T0 = (-b - bb4ac)  / (2.0 * a);
	T1 = (-b + bb4ac)  / (2.0 * a);

	if(T0 < 0)
	{
		if(T1 < 0)
		{
			return false; //T0,T1 behind camera
		}
		t = T1;
		return true;
	}
	if(T1 < 0)
	{
		if(T0 < 0)
		{
			return false; //behind camera
		}
		t = T0;
		return true;
	}

	if(T0 < T1)
	{
		t = T0;
		return true;
	}

	t = T1;
	return true;
}

bool Sphere::contains(const Vector& v, double expand)
{
	return (v-pos()).length() < (radius() + expand);
}


bool Sphere::overlapRadius(Sphere& s2, double m)
{
	return (*pos() - *s2.pos()).length() < ((radius() + s2.radius())*m);	
}

bool Sphere::overlapRadius(Sphere* s2, double m)
{
	return overlapRadius(*s2, m);
}

double Sphere::volume()
{
	return 4.0/3.0 * 3.14159265358979 * pow(radius(), 3.0);
}



static int l_setpos(lua_State* L)
{
	LUA_PREAMBLE(Sphere, s, 1);
	Vector p;
	lua_makevector(L, 2, p);
	s->setPos(&p);
	return 0;
}
static int l_getpos(lua_State* L)
{
	LUA_PREAMBLE(Sphere, s, 1);
	luaT_push<Vector>(L, s->pos());
	return 1;
}
static int l_setrad(lua_State* L)
{
	LUA_PREAMBLE(Sphere, s, 1);
	s->setRadius(lua_tonumber(L, 1));
	return 0;
}
static int l_getrad(lua_State* L)
{
	LUA_PREAMBLE(Sphere, s, 1);
	lua_pushnumber(L, s->radius());
	return 1;
}
static int l_overlapRadius(lua_State* L)
{
	LUA_PREAMBLE(Sphere, s1, 1);
	LUA_PREAMBLE(Sphere, s2, 2);

	double m = 1.0;
	if(lua_isnumber(L, 3))
		m =lua_tonumber(L, 4);
	
	lua_pushboolean(L, s1->overlapRadius(s2, m));
	return 1;
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Sphere::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, Volume::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"setPosition", l_setpos},
		{"position",    l_getpos},
		{"setRadius", l_setrad},
		{"radius",    l_getrad},
		{"overlapRadius", l_overlapRadius},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}

