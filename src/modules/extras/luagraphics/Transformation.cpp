#include <math.h>
#include "Matrix.h"
#include "Vector.h"
#include <stdlib.h>
#include "Transformation.h"

Transformation::Transformation()
	: LuaBaseObject(hash32("Transformation"))
{
	nextTransform = 0;
	values[0] = 0;
	values[1] = 0;
	values[2] = 0;
	type = none;
}

Transformation::Transformation(int etype)
	: LuaBaseObject(etype)
{
	nextTransform = 0;
	values[0] = 0;
	values[1] = 0;
	values[2] = 0;
	type = none;
}

Transformation::~Transformation()
{
	for(unsigned int i=0; i<volumes.size(); i++)
	{
		luaT_dec<Volume>(volumes[i]);
	}
	luaT_dec<Transformation>(nextTransform);
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

void Transformation::setNextTransformation(Transformation* nt)
{
	luaT_inc<Transformation>(nt);
	luaT_dec<Transformation>(nextTransform);
	nextTransform = nt;
}

void Transformation::addVolume(Volume* v)
{
	luaT_inc<Volume>(v);
	volumes.push_back(v);
}




static int l_add(lua_State* L)
{
	LUA_PREAMBLE(Transformation, t, 1);
	if(luaT_is<Volume>(L, 2))
	{
		t->addVolume(luaT_to<Volume>(L, 2));
		return 0;
	}
	if(luaT_is<Transformation>(L, 2))
	{
		t->setNextTransformation(luaT_to<Transformation>(L, 2));
		return 0;
	}
	return 0;
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

