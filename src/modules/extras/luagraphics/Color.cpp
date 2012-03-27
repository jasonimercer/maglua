#include "Color.h"

Color::Color(double r, double g, double b, double a)
{
	rgba[0] = r;
	rgba[1] = g;
	rgba[2] = b;
	rgba[3] = a;
}


int Color::luaInit(lua_State* L)
{
	for(int i=0; i<4; i++)
	{
		if(lua_isnumber(L, i+1))
			rgba[i] = lua_tonumber(L, i+1);
	}
	return 0;
}

void Color::push(lua_State* L)
{
	luaT_push<Color>(L, this);
}


void Color::set(Color* c)
{
	if(!c) return;
	(*this) = *c;
}


void  Color::setComponent(int i, double v)
{
	if(i < 0) i = 0;	if(i > 3) i = 3;
	if(v < 0) v = 0;	if(v > 1) v = 1;
	rgba[i] = v;
}

double Color::component(int i) const
{
	if(i < 0) i = 0;
	if(i > 3) i = 3;
	return rgba[i];
}

Color& Color::operator =(const Color& rhs)
{
	for(int i=0; i<4; i++)
		setComponent(i, rhs.component(i));
}

Color& Color::operator =(const Vector& rhs)
{
	rgba[0] = rhs.x();
	rgba[1] = rhs.y();
	rgba[2] = rhs.z();
	rgba[3] = 1;
}


LUAFUNC_GET_DOUBLE(Color, rgba[0], l_gr);
LUAFUNC_GET_DOUBLE(Color, rgba[1], l_gg);
LUAFUNC_GET_DOUBLE(Color, rgba[2], l_gb);
LUAFUNC_GET_DOUBLE(Color, rgba[3], l_ga);

LUAFUNC_SET_DOUBLE(Color, rgba[0], l_sr);
LUAFUNC_SET_DOUBLE(Color, rgba[1], l_sg);
LUAFUNC_SET_DOUBLE(Color, rgba[2], l_sb);
LUAFUNC_SET_DOUBLE(Color, rgba[3], l_sa);

static int l_set(lua_State* L)
{
	LUA_PREAMBLE(Color, c, 1);
	
	if(luaT_is<Color>(L, 2))
	{
		c->set(luaT_to<Color>(L, 2));
	}
	else
	{
		for(int i=0; i<4; i++)
		{
			if(lua_isnumber(L, i+2))
				c->rgba[i] = lua_tonumber(L, i+2);
		}		
	}
}

static int l_get(lua_State* L)
{
	LUA_PREAMBLE(Color, c, 1);
	for(int i=0; i<4; i++)
		lua_pushnumber(L, c->rgba[i]);
	return 4;
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Color::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LuaBaseObject::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"red",     l_gr},
		{"green",   l_gg},
		{"blue",    l_gb},
		{"alpha",   l_ga},
		{"setRed",     l_sr},
		{"setGreen",   l_sg},
		{"setBlue",    l_sb},
		{"setAlpha",   l_sa},
		{"set", l_set},
		{"get", l_get},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}

