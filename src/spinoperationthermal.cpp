/******************************************************************************
* Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

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
	markSlotUsed(ss);

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

static int l_thermal_tostring(lua_State* L)
{
	Thermal* th = checkThermal(L, 1);
	if(!th) return 0;
	
	lua_pushfstring(L, "Thermal (%dx%dx%d)", th->nx, th->ny, th->nz);
	
	return 1;
}


static int l_thermal_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.thermal");
	return 1;
}

static int l_thermal_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Generates a the random thermal field of a *SpinSystem*");
		lua_pushstring(L, ""); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	if(lua_istable(L, 1))
	{
		return 0;
	}
	
	if(!lua_iscfunction(L, 1))
	{
		return luaL_error(L, "help expect zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if(func == l_thermal_new)
	{
		lua_pushstring(L, "Create a new Thermal Operator.");
		lua_pushstring(L, "1 *3Vector*: system size"); 
		lua_pushstring(L, "1 Thermal object");
		return 3;
	}
	
	if(func == l_thermal_apply)
	{
		lua_pushstring(L, "Generates a the random thermal field of a *SpinSystem*");
		lua_pushstring(L, "1 *Random*, 1 *SpinSystem*: The first argument is a random number generator that is used as a source of random values. The second argument is the spin system which will receive the field.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_thermal_scalesite)
	{
		lua_pushstring(L, "Scale the thermal field at a site. This allows non-uniform thermal effects over a lattice.");
		lua_pushstring(L, "1 *3Vector*, 1 Number: The vectors define the lattice sites that will have a scaled thermal effect, the number is the how the thermal field is scaled.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_thermal_settemp)
	{
		lua_pushstring(L, "Sets the base value of the temperature. ");
		lua_pushstring(L, "1 number: temperature of the system.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_thermal_gettemp)
	{
		lua_pushstring(L, "Gets the base value of the temperature. ");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 number: temperature of the system.");
		return 3;
	}
	
	
	if(func == l_thermal_member)
	{
		lua_pushstring(L, "Determine if a lattice site is a member of the Operation.");
		lua_pushstring(L, "3 Integers: lattics site x, y, z.");
		lua_pushstring(L, "1 Boolean: True if x, y, z is part of the Operation, False otherwise.");
		return 3;
	}
	
	return 0;
}


void registerThermal(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_thermal_gc},
		{"__tostring",   l_thermal_tostring},
		{"apply",        l_thermal_apply},
		{"member",       l_thermal_member},
		{"scaleSite",    l_thermal_scalesite},
		{"setTemperature", l_thermal_settemp},
		{"set",          l_thermal_settemp},
		{"get",          l_thermal_gettemp},
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
		{"help",                l_thermal_help},
		{"metatable",           l_thermal_mt},
		{NULL, NULL}
	};
		
	luaL_register(L, "Thermal", functions);
	lua_pop(L,1);	
}

