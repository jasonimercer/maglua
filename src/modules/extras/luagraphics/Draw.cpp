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

	for(int i=2; i<=lua_gettop(L); i++)
	{
		if(luaT_is<Light>(L, i))
		{
			d->draw(* luaT_to<Light>(L, i));
			continue;
		}
		if(luaT_is<Sphere>(L, i))
		{
			d->draw(* luaT_to<Sphere>(L, i));
			continue;
		}
		if(luaT_is<Group>(L, i))
		{
			d->draw(* luaT_to<Group>(L, i));
			continue;
		}
		if(luaT_is<Camera>(L, i))
		{
			d->draw(* luaT_to<Camera>(L, i));
			continue;
		}
		if(luaT_is<Tube>(L, i))
		{
			d->draw(* luaT_to<Tube>(L, i));
			continue;
		}
		if(luaT_is<Transformation>(L, i))
		{
			d->draw(* luaT_to<Transformation>(L, i));
			continue;
		}
		return luaL_error(L, "Failed to draw object");
	}
	return 0;
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
