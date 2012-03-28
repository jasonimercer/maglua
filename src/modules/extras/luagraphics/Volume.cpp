#include "Volume.h"
#include "AABB.h"

Volume::Volume(int etype)
	: LuaBaseObject(etype)
{
	bb = luaT_inc<AABB>(new AABB);
	selected = false;
	color = luaT_inc<Color>(new Color);
}

Volume::~Volume()
{
	luaT_dec<AABB>(bb);
	luaT_dec<Color>(color);
}

int Volume::luaInit(lua_State* L)
{
	return 0;
}

void Volume::push(lua_State* L)
{
	luaT_push<Volume>(L, this);
}


AABB* Volume::getBB()
{
	return bb;
}

void Volume::setColor(Color* c)
{
	luaT_inc<Color>(c);
	luaT_dec<Color>(color);
	color = c;
}



	
static int l_volume(lua_State* L)
{
	LUA_PREAMBLE(Volume, v, 1);
	lua_pushnumber(L, v->volume());
	return 1;
}

static int l_rayisect(lua_State* L)
{
	LUA_PREAMBLE(Volume, v, 1);
	LUA_PREAMBLE(Ray, r, 1);
	double t;
	bool b = v->rayIntersect(*r, t);
	lua_pushboolean(L, b);
	lua_pushnumber(L, t);
	return 2;	
}

static int l_contains(lua_State* L)
{
	LUA_PREAMBLE(Volume, v, 1);
	LUA_PREAMBLE(Vector, vec, 2);
	double expand = 0;
	if(lua_isnumber(L, 3))
		expand = lua_tonumber(L, 3);
	
	lua_pushboolean(L, v->contains(*vec, expand));
	return 1;
}

static int l_excludes(lua_State* L)
{
	LUA_PREAMBLE(Volume, v, 1);
	LUA_PREAMBLE(Vector, vec, 2);
	double expand = 0;
	if(lua_isnumber(L, 3))
		expand = lua_tonumber(L, 3);
	
	lua_pushboolean(L, v->excludes(*vec, expand));
	return 1;
}

static int l_getbb(lua_State* L)
{
	LUA_PREAMBLE(Volume, v, 1);
	luaT_push<AABB>(L, v->getBB());
	return 1;
}

static int l_setcolor(lua_State* L)
{
	LUA_PREAMBLE(Volume, v, 1);
	LUA_PREAMBLE(Color, col, 2);

	v->setColor(col);
	return 0;
}

static int l_getcolor(lua_State* L)
{
	LUA_PREAMBLE(Volume, v, 1);

	luaT_push<Color>(L, v->color);
	return 1;
}
	
static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Volume::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LuaBaseObject::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"setColor", l_setcolor},
		{"color", l_getcolor},
		{"volume",       l_volume},
		{"rayIntersect", l_rayisect},
		{"contains",     l_contains},
		{"excludes",     l_excludes},
		{"getBB",        l_getbb},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}