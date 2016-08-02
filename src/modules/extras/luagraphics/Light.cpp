#include "Light.h"

Light::Light()
: Sphere(hash32("Light"))
{
	diffuse_color = luaT_inc<Color>(new Color(0.8,0.8,0.8));
	specular_color = luaT_inc<Color>(new Color(1,1,1));
}


int Light::luaInit(lua_State* L)
{
	return Sphere::luaInit(L);
}

void Light::push(lua_State* L)
{
	luaT_push<Light>(L, this);
}




static int l_light_setdcolor(lua_State* L)
{
	LUA_PREAMBLE(Light, a, 1);	
	a->diffuse_color->set(luaT_to<Color>(L, 2));
	return 0;
}

static int l_light_setscolor(lua_State* L)
{
	LUA_PREAMBLE(Light, a, 1);
	a->specular_color->set(luaT_to<Color>(L, 2));
	return 0;
}

static int l_light_getdcolor(lua_State* L)
{
	LUA_PREAMBLE(Light, a, 1);	
	luaT_push<Color>(L, a->diffuse_color);
	return 1;
}

static int l_light_getscolor(lua_State* L)
{
	LUA_PREAMBLE(Light, a, 1);	
	luaT_push<Color>(L, a->specular_color);
	return 1;
}



static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Light::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, Sphere::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"setSpecularColor",     l_light_setscolor},
		{"specularColor",        l_light_getscolor},
		
		{"setDiffuseColor",     l_light_setdcolor},
		{"diffuseColor",        l_light_getdcolor},

		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}
