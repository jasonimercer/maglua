#include "spinoperationthermal.h"
#include "spinsystem.h"
#include "random.h"
#include "llg.h"
#include "llgquat.h"
#include <math.h>
#include <stdio.h>

Thermal::Thermal(int nx, int ny, int nz)
	: SpinOperation("Thermal", THERMAL_SLOT, nx, ny, nz, ENCODE_THERMAL)
{
	scale = new double[nxyz];
	for(int i=0; i<nxyz; i++)
		scale[i] = 1.0;
	temperature = 0;
}

void Thermal::encode(buffer* b) const
{
	encodeInteger(nx, b);
	encodeInteger(ny, b);
	encodeInteger(nz, b);
	
	encodeDouble(temperature, b);
	
	for(int i=0; i<nxyz; i++)
		encodeDouble(scale[i], b);
}

int  Thermal::decode(buffer* b)
{
	nx = decodeInteger(b);
	ny = decodeInteger(b);
	nz = decodeInteger(b);
	nxyz = nx * ny * nz;

	temperature = decodeDouble(b);
	
	if(scale)
		delete [] scale;
	
	scale = new double[nxyz];
	
	for(int i=0; i<nxyz; i++)
		scale[i] = decodeDouble(b);
	return 0;
}

Thermal::~Thermal()
{
	delete [] scale;
}

bool Thermal::apply(RNG* rand, SpinSystem* ss)
{
	const double alpha = ss->alpha;
	const double dt    = ss->dt;
	const double gamma = ss->gamma;

	double* hx = ss->hx[slot];
	double* hy = ss->hy[slot];
	double* hz = ss->hz[slot];
	
	for(int i=0; i<ss->nxyz; i++)
	{
		const double ms = ss->ms[i];
		if(ms != 0 && temperature != 0)
		{
// 			double stddev = sqrt((2.0 * alpha * temperature * scale[i]) / (ms * dt * gamma * (1+alpha*alpha)));
			const double stddev = sqrt((2.0 * alpha * temperature * scale[i]) / (ms * dt * gamma));
			
			hx[i] = stddev * rand->randNorm(0, 1);
			hy[i] = stddev * rand->randNorm(0, 1);
			hz[i] = stddev * rand->randNorm(0, 1);
		}
		else
		{
			hx[i] = 0;
			hy[i] = 0;
			hz[i] = 0;
		}
	}
	return true;
}

void Thermal::scaleSite(int px, int py, int pz, double strength)
{
	if(member(px, py, pz))
	{
		int idx = getidx(px, py, pz);
		scale[idx] = strength;
	}
}



Thermal* checkThermal(lua_State* L, int idx)
{
	Thermal** pp = (Thermal**)luaL_checkudata(L, idx, "MERCER.thermal");
    luaL_argcheck(L, pp != NULL, 1, "`Thermal' expected");
    return *pp;
}

void lua_pushThermal(lua_State* L, Thermal* th)
{
	th->refcount++;
	
	Thermal** pp = (Thermal**)lua_newuserdata(L, sizeof(Thermal**));
	
	*pp = th;
	luaL_getmetatable(L, "MERCER.thermal");
	lua_setmetatable(L, -2);
}

int l_thermal_new(lua_State* L)
{
	int n[3];
	lua_getnewargs(L, n, 1);

	lua_pushThermal(L, new Thermal(n[0], n[1], n[2]));
	return 1;
}

int l_thermal_gc(lua_State* L)
{
	Thermal* th = checkThermal(L, 1);
	if(!th) return 0;
	
	th->refcount--;
	if(th->refcount == 0)
		delete th;
	
	return 0;
}

int l_thermal_apply(lua_State* L)
{
	Thermal*    th = checkThermal(L, 1);
	//LLG*       llg = checkLLG(L, 2);
	RNG*       rng = checkRandom(L, 3-1);
	SpinSystem* ss = checkSpinSystem(L, 4-1);

	if(!th)
		return luaL_error(L, "Thermal object required");

// 	if(!rng || !ss || !llg)
// 		return luaL_error(L, "Thermal.apply requires llg, rng, spinsystem");
	
	if(!rng || !ss)
		return luaL_error(L, "Thermal.apply requires rng, spinsystem");
	
	if(!th->apply(rng,ss))
		return luaL_error(L, th->errormsg.c_str());
	
	return 0;
}

int l_thermal_scalesite(lua_State* L)
{
	Thermal* th = checkThermal(L, 1);
	if(!th) return 0;

	int s[3];
	int r = lua_getNint(L, 3, s, 2, 1);
	if(r<0)
		return luaL_error(L, "invalid site");
	
	double v = lua_tonumber(L, 2+r);
	
	th->scaleSite(
		s[0] - 1,
		s[1] - 1,
		s[2] - 1,
		v);

	return 0;
}

int l_thermal_settemp(lua_State* L)
{
	Thermal* th = checkThermal(L, 1);
	if(!th) return 0;

	if(lua_isnil(L, 2))
		return luaL_error(L, "set temp cannot be nil");

	th->temperature = lua_tonumber(L, 2);
	return 0;
}
int l_thermal_gettemp(lua_State* L)
{
	Thermal* th = checkThermal(L, 1);
	if(!th) return 0;

	lua_pushnumber(L, th->temperature);
	return 1;
}

int l_thermal_member(lua_State* L)
{
	Thermal* th = checkThermal(L, 1);
	if(!th) return 0;

	int px = lua_tointeger(L, 2) - 1;
	int py = lua_tointeger(L, 3) - 1;
	int pz = lua_tointeger(L, 4) - 1;
	
	if(th->member(px, py, pz))
		lua_pushboolean(L, 1);
	else
		lua_pushboolean(L, 0);

	return 1;
}

void registerThermal(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_thermal_gc},
		{"apply",        l_thermal_apply},
		{"member",       l_thermal_member},
		{"scaleSite",    l_thermal_scalesite},
		{"setTemperature", l_thermal_settemp},
		{"temperature",  l_thermal_gettemp},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.thermal");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_thermal_new},
		{NULL, NULL}
	};
		
	luaL_register(L, "Thermal", functions);
	lua_pop(L,1);	
}

