#include "spinoperationanisotropy.h"
#include "spinsystem.h"

#include <stdlib.h>

Anisotropy::Anisotropy(int nx, int ny, int nz)
	: SpinOperation("Anisotropy", ANISOTROPY_SLOT, nx, ny, nz)
{
	ax = new double[nxyz];
	ay = new double[nxyz];
	az = new double[nxyz];
	strength = new double[nxyz];

	for(int i=0; i<nxyz; i++)
	{
		ax[i] = 0;
		ay[i] = 0;
		az[i] = 1;

		strength[i] = 1;
	}
}

Anisotropy::~Anisotropy()
{
	delete [] ax;
	delete [] ay;
	delete [] az;
	delete [] strength;
}

bool Anisotropy::apply(SpinSystem* ss)
{
	double SpinDotEasyAxis;
	double v;
	for(int i=0; i<nxyz; i++)
	{
		const double ms = ss->ms[i];
		if(ms > 0)
		{
			SpinDotEasyAxis = ss->x[i] * ax[i] +
			                  ss->y[i] * ay[i] +
			                  ss->z[i] * az[i];

			v = strength[i] * SpinDotEasyAxis / (ms * ms);

			ss->hx[slot][i] = ax[i] * v;
			ss->hy[slot][i] = ay[i] * v;
			ss->hz[slot][i] = az[i] * v;
		}
	}
	return true;
}







Anisotropy* checkAnisotropy(lua_State* L, int idx)
{
	Anisotropy** pp = (Anisotropy**)luaL_checkudata(L, idx, "MERCER.anisotropy");
    luaL_argcheck(L, pp != NULL, 1, "`Anisotropy' expected");
    return *pp;
}


int l_ani_new(lua_State* L)
{
	if(lua_gettop(L) != 3)
		return luaL_error(L, "Anisotropy.new requires nx, ny, nz");

	Anisotropy* ani = new Anisotropy(
			lua_tointeger(L, 1),
			lua_tointeger(L, 2),
			lua_tointeger(L, 3)
	);
	ani->refcount++;
	
	Anisotropy** pp = (Anisotropy**)lua_newuserdata(L, sizeof(Anisotropy**));
	
	*pp = ani;
	luaL_getmetatable(L, "MERCER.anisotropy");
	lua_setmetatable(L, -2);
	return 1;
}

int l_ani_gc(lua_State* L)
{
	Anisotropy* ani = checkAnisotropy(L, 1);
	if(!ani) return 0;
	
	ani->refcount--;
	if(ani->refcount == 0)
		delete ani;
	
	return 0;
}

int l_ani_apply(lua_State* L)
{
	Anisotropy* ani = checkAnisotropy(L, 1);
	SpinSystem* ss = checkSpinSystem(L, 2);
	
	if(!ani->apply(ss))
		return luaL_error(L, ani->errormsg.c_str());
	
	return 0;
}

int l_ani_member(lua_State* L)
{
	Anisotropy* ani = checkAnisotropy(L, 1);
	if(!ani) return 0;

	int px = lua_tointeger(L, 2) - 1;
	int py = lua_tointeger(L, 3) - 1;
	int pz = lua_tointeger(L, 4) - 1;
	
	if(ani->member(px, py, pz))
		lua_pushboolean(L, 1);
	else
		lua_pushboolean(L, 0);

	return 1;
}

int l_ani_set(lua_State* L)
{
	Anisotropy* ani = checkAnisotropy(L, 1);
	if(!ani) return 0;

	int px = lua_tointeger(L, 2) - 1;
	int py = lua_tointeger(L, 3) - 1;
	int pz = lua_tointeger(L, 4) - 1;

	if(!ani->member(px, py, pz))
		return luaL_error(L, "site is not part of system");

	int idx = ani->getidx(px, py, pz);

	ani->ax[idx] = lua_tonumber(L, 5);
	ani->ay[idx] = lua_tonumber(L, 6);
	ani->az[idx] = lua_tonumber(L, 7);

	/* anisotropy axis is a unit vector */
	const double a = 
		ani->ax[idx]*ani->ax[idx] +
		ani->ay[idx]*ani->ay[idx] +
		ani->az[idx]*ani->az[idx];
	
	if(a > 0)
	{
		ani->ax[idx] /= sqrt(a);
		ani->ay[idx] /= sqrt(a);
		ani->az[idx] /= sqrt(a);
	}
	// else leave it as zero

	ani->strength[idx] = lua_tonumber(L, 8);

	return 0;
}


void registerAnisotropy(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_ani_gc},
		{"apply",        l_ani_apply},
		{"setSite",      l_ani_set},
		{"member",       l_ani_member},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.anisotropy");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_ani_new},
		{NULL, NULL}
	};
		
	luaL_register(L, "Anisotropy", functions);
	lua_pop(L,1);	
}

