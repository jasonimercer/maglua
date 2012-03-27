#include "Atom.h"

Atom::Atom(lua_State* _L)
	: Sphere()
{
	occupancy = 1.0;
	vdwRadius = 0.0;
	dataRef = -1;
	L =_L;
}


Atom::Atom(const Atom& other)
	: Sphere()
{
	setPos(other.pos());
	setRadius(other.radius());
	name = other.name;
	type = other.type;
	occupancy = other.occupancy;
	color = other.color;
	vdwRadius = other.vdwRadius;
	dataRef = -1;
	L = other.L;
}

Atom::Atom(lua_State* _L, const Sphere& shape, string n, string t)
	: Sphere(shape)
{
	name = n;
	type = t;
	occupancy = 1;
	refcount = 0;
	vdwRadius = 0.0;
	dataRef = -1;
	L = _L;
}

// void Atom::draw(Draw& drawfuncs) const
// {
// 	drawfuncs.draw(*this, color);
// }

Atom::~Atom()
{
	if(dataRef >= 0)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, dataRef);
	}
}





int lua_isatom(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "Atom");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

Atom* lua_toatom(lua_State* L, int idx)
{
	Atom** pp = (Atom**)luaL_checkudata(L, idx, "Atom");
	luaL_argcheck(L, pp != NULL, idx, "`Atom' expected");
	return *pp;
}

void lua_pushatom(lua_State* L, Atom* a)
{
	Atom** pp = (Atom**)lua_newuserdata(L, sizeof(Atom**));

	*pp = a;
	luaL_getmetatable(L, "Atom");
	lua_setmetatable(L, -2);
	a->refcount++;
}

static int l_atom_new(lua_State* L)
{
	lua_pushatom(L, new Atom(L));
	return 1;
}

static int l_atom_gc(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;
	
	a->refcount--;
	if(a->refcount == 0)
		delete a;
	return 0;
}
static int l_atom_eq(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;

	Atom* b = lua_toatom(L, 2);
	if(!b) return 0;

	lua_pushboolean(L, a==b);
	return 1;
}

static int l_atom_getpos(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;

	lua_pushvector(L, new Vector(a->pos()));
	return 1;
}

static int l_atom_setpos(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;

	Vector v;
	lua_makevector(L, 2, v);
	
	a->setPos(v);
}

static int l_atom_setcolor(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;

	if(lua_istable(L, 2))
	{
		for(int i=0; i<4; i++)
		{
			double c = 1.0;
			lua_pushinteger(L, i+1);
			lua_gettable(L, 2);
			if(lua_isnumber(L, -1))
				c = lua_tonumber(L, -1);
			lua_pop(L, 1);
			a->color.setComponent(i, c);
		}
		return 0;
	}
	if(lua_isvector(L, 2))
	{
		Vector v;
		lua_makevector(L, 2, v);
		a->color = v;
		return 0;
	}
	for(int i=0; i<4; i++)
	{
		double c = 1.0;
		if(lua_isnumber(L, i+2))
			c = lua_tonumber(L, i+2);
		a->color.setComponent(i, c);
	}
		
	return 0;
}

static int l_atom_getcolor(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;

	for(int i=0; i<4; i++)
	{
		lua_pushnumber(L, a->color.component(i));
	}
	return 4;
}

static int l_atom_setradius(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;
	a->setRadius(lua_tonumber(L, 2));
	return 0;
}
static int l_atom_getradius(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;
	lua_pushnumber(L, a->radius());
	return 1;
}


static int l_atom_setvdwradius(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;
	a->vdwRadius = lua_tonumber(L, 2);
	return 0;
}
static int l_atom_getvdwradius(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;
	lua_pushnumber(L, a->vdwRadius);
	return 1;
}

static int l_atom_setname(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;
	a->name = lua_tostring(L, 2);
	return 0;
}
static int l_atom_getname(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;
	lua_pushstring(L, a->name.c_str());
	return 1;
}

static int l_atom_settype(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;
	a->type = lua_tostring(L, 2);
	return 0;
}
static int l_atom_gettype(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;
	lua_pushstring(L, a->type.c_str());
	return 1;
}



static int l_atom_setselected(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;
	a->selected = lua_toboolean(L, 2);
	return 0;
}
static int l_atom_getselected(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;
	lua_pushboolean(L, a->selected);
	return 1;
}

static int l_atom_rayintersect(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;
	
	Ray ray;
	lua_makeray(L, 2, ray);

	double t;
	
	bool b = a->rayIntersect(ray, t);
	
	lua_pushboolean(L, b);
	lua_pushnumber(L, t);
	return 2;
}

static int l_atom_setdata(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;
	
	if(a->dataRef >= 0)
	{
		luaL_unref(L, LUA_REGISTRYINDEX, a->dataRef);
	}
	if(lua_isnil(L, -1))
	{
		a->dataRef = -1;
	}
	else
	{
		a->dataRef = luaL_ref(L, LUA_REGISTRYINDEX);
	}
	return 0;
}

static int l_atom_getdata(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;
	
	if(a->dataRef < 0)
	{
		lua_pushnil(L);
	}
	else
	{
		lua_rawgeti(L, LUA_REGISTRYINDEX, a->dataRef);
	}
	return 1;
}

static int l_tostring(lua_State* L)
{
	Atom* a = lua_toatom(L, 1);
	if(!a) return 0;
	
	lua_pushfstring(L, "Atom (%s, %s)", a->type.c_str(), a->name.c_str());
	return 1;
}



void lua_registeratom(lua_State* L)
{
	static const struct luaL_reg struct_m [] = { //methods
		{"__gc",         l_atom_gc},
		{"__eq",         l_atom_eq},
		{"__tostring",   l_tostring},

		{"pos",          l_atom_getpos},
		{"setPos",       l_atom_setpos},

 		{"setColor",     l_atom_setcolor},
 		{"color",        l_atom_getcolor},
// 		{"inLayer",      l_atom_getinlayer},
// 		{"setInLayer",   l_atom_setinlayer},
// 		{"setSelected",  l_atom_setselected},
// 		{"selected",     l_atom_getselected},
// 		{"overlapVDW",   l_atom_overlapvdw},
		{"name",         l_atom_getname},
		{"setName",      l_atom_setname},

		{"type",         l_atom_gettype},
		{"setType",      l_atom_settype},
// 		{"visible",      l_atom_isvisible},
// 		{"resetPosition", l_atom_resetposition},
// 		{"rotate",       l_atom_rotate},
		{"radius",       l_atom_getradius},
		{"setRadius",    l_atom_setradius},

		{"vdwRadius",   l_atom_getvdwradius},
		{"setVdwRadius",l_atom_setvdwradius},
// 		{"occupancy",    l_atom_getoccupancy},
// 		{"setOccupancy", l_atom_setoccupancy},
		{"selected",     l_atom_getselected},
		{"setSelected",  l_atom_setselected},
		
		{"rayIntersect", l_atom_rayintersect},
		
		{"data",         l_atom_getdata},
		{"setData",      l_atom_setdata},
		{NULL, NULL}
	};

	luaL_newmetatable(L, "Atom");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, struct_m);
	lua_pop(L,1); //metatable is registered

	static const struct luaL_reg struct_f [] = {
			{"new", l_atom_new},
			{NULL, NULL}
	};

	luaL_register(L, "Atom", struct_f);
	lua_pop(L,1);
}

