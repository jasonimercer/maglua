#include <math.h>
#include "Matrix.h"
#include "Vector.h"
#include <stdlib.h>
#include "Transformation.h"

Transformation::Transformation()
	: LuaBaseObject(hash32("Transformation"))
{
	values[0] = 0;
	values[1] = 0;
	values[2] = 0;

	transformations.clear();
	volumes.clear();
}

Transformation::Transformation(int etype)
	: LuaBaseObject(etype)
{
	values[0] = 0;
	values[1] = 0;
	values[2] = 0;

	transformations.clear();
	volumes.clear();
}

Transformation::~Transformation()
{
	for(unsigned int i=0; i<transformations.size(); i++)
		luaT_dec<Transformation>(transformations[i]);
	
	for(unsigned int i=0; i<volumes.size(); i++)
		luaT_dec<Volume>(volumes[i]);
}

int Transformation::luaInit(lua_State* L)
{
	for(int i=0; i<3; i++)
	{
		values[i] = lua_tonumber(L, i+1);
	}
	return 0;
}

void Transformation::push(lua_State* L)
{
	luaT_push<Transformation>(L, this);
}

void Transformation::addTransformation(Transformation* t)
{
	luaT_inc<Transformation>(t);
	transformations.push_back(t);
}

void Transformation::addVolume(Volume* v)
{
	luaT_inc<Volume>(v);
	volumes.push_back(v);
}


void Transformation::removeTransformation(Transformation* t)
{
	vector<Transformation*>::iterator it;
	for(it=transformations.begin(); it!=transformations.end(); ++it)
	{
		if(*it == t)
		{
			luaT_dec<Transformation>(t);
			transformations.erase(it);
			return removeTransformation(t);
		}
	}
}


void Transformation::removeVolume(Volume* t)
{
	vector<Volume*>::iterator it;
	for(it=volumes.begin(); it!=volumes.end(); ++it)
	{
		if(*it == t)
		{
			luaT_dec<Volume>(t);
			volumes.erase(it);
			return removeVolume(t);
		}
	}
}

static int l_add(lua_State* L)
{
	LUA_PREAMBLE(Transformation, t, 1);
	for(int i=2; i<=lua_gettop(L); i++)
	{
		if(luaT_is<Volume>(L, i))
		{
			t->addVolume(luaT_to<Volume>(L, i));
		}
		if(luaT_is<Transformation>(L, i))
		{
			t->addTransformation(luaT_to<Transformation>(L, i));
		}
	}
	return 0;
}

static int l_remove(lua_State* L)
{
	LUA_PREAMBLE(Transformation, t, 1);
	for(int i=2; i<=lua_gettop(L); i++)
	{
		if(luaT_is<Volume>(L, i))
		{
			t->removeVolume(luaT_to<Volume>(L, i));
		}
		if(luaT_is<Transformation>(L, i))
		{
			t->removeTransformation(luaT_to<Transformation>(L, i));
		}
	}
	return 0;
}

static int l_set(lua_State* L)
{
	LUA_PREAMBLE(Transformation, t, 1);
	for(int i=0; i<3; i++)
		t->values[i] = lua_tonumber(L, i+2);
	return 0;
}
static int l_get(lua_State* L)
{
	LUA_PREAMBLE(Transformation, t, 1);
	for(int i=0; i<3; i++)
		lua_pushnumber(L, t->values[i]);
	return 3;
}

LUAFUNC_SET_DOUBLE(Transformation, values[0], l_setX)
LUAFUNC_SET_DOUBLE(Transformation, values[1], l_setY)
LUAFUNC_SET_DOUBLE(Transformation, values[2], l_setZ)

LUAFUNC_GET_DOUBLE(Transformation, values[0], l_getX)
LUAFUNC_GET_DOUBLE(Transformation, values[1], l_getY)
LUAFUNC_GET_DOUBLE(Transformation, values[2], l_getZ)

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Transformation::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LuaBaseObject::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"add",      l_add},
		{"remove",   l_remove},
		{"set",      l_set},
		{"get",      l_get},
		{"setX",     l_setX},
		{"setY",     l_setY},
		{"setZ",     l_setZ},
		{"x",     l_getX},
		{"y",     l_getY},
		{"z",     l_getZ},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}

