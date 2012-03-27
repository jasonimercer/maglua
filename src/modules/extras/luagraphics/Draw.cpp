#include "Draw.h"


static int l_reset(lua_State* L)
{
	LUA_PREAMBLE(Draw, d, 1);
	d->reset();
	return 0;
}

static int l_draw(lua_State* L)
{
	LUA_PREAMBLE(Draw, d, 1);


	if(luaT_is<Light>(L, 2))
	{
		d->draw(* luaT_to<Light>(L, 2));
		return 0;
	}

	if(luaT_is<Sphere>(L, 2))
	{
		d->draw(* luaT_to<Sphere>(L, 2));
		return 0;
	}

	if(luaT_is<Group>(L, 2))
	{
		d->draw(* luaT_to<Group>(L, 2));
		return 0;
	}

	if(luaT_is<Camera>(L, 2))
	{
		d->draw(* luaT_to<Camera>(L, 2));
		return 0;
	}



	if(luaT_is<Tube>(L, 2))
	{
		d->draw(* luaT_to<Tube>(L, 2));
		return 0;
	}

/*
	if(lua_isvolumelua(L, 2))
	{
		d->draw(*lua_tovolumelua(L, 2));
		return 0;
	}
*/	
	return luaL_error(L, "Failed to draw object");
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Draw::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LuaBaseObject::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"draw",     l_draw},
		{"reset",     l_reset},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}
