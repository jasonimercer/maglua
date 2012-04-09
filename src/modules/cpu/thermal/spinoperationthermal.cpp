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
#include <math.h>
#include <stdio.h>

Thermal::Thermal(int nx, int ny, int nz)
	: SpinOperation(Thermal::typeName(), THERMAL_SLOT, nx, ny, nz, hash32(Thermal::typeName()))
{
	scale = new double[nxyz];
	for(int i=0; i<nxyz; i++)
		scale[i] = 1.0;
	temperature = 0;
}

int Thermal::luaInit(lua_State* L)
{
	if(scale)
		delete [] scale;
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz

	scale = new double[nxyz];
	for(int i=0; i<nxyz; i++)
		scale[i] = 1.0;
	temperature = 0;
	return 0;
}

void Thermal::push(lua_State* L)
{
	luaT_push<Thermal>(L, this);
}

void Thermal::encode(buffer* b)
{
	SpinOperation::encode(b);
	
	encodeDouble(temperature, b);
	
	for(int i=0; i<nxyz; i++)
		encodeDouble(scale[i], b);
}

int  Thermal::decode(buffer* b)
{
	SpinOperation::decode(b);
	
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
			const double stddev = global_scale * sqrt((2.0 * alpha * temperature * scale[i]) / (ms * dt * gamma));
			
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



static int l_apply(lua_State* L)
{
	LUA_PREAMBLE(Thermal, th, 1);
	LUA_PREAMBLE(RNG,    rng, 2);
	LUA_PREAMBLE(SpinSystem,ss,3);

	if(!th->apply(rng,ss))
		return luaL_error(L, th->errormsg.c_str());
	
	return 0;
}

static int l_scalesite(lua_State* L)
{
	LUA_PREAMBLE(Thermal, th, 1);

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

static int l_settemp(lua_State* L)
{
	LUA_PREAMBLE(Thermal, th, 1);

	if(lua_isnil(L, 2))
		return luaL_error(L, "set temp cannot be nil");

	th->temperature = lua_tonumber(L, 2);
	return 0;
}
static int l_gettemp(lua_State* L)
{
	LUA_PREAMBLE(Thermal, th, 1);

	lua_pushnumber(L, th->temperature);
	return 1;
}

int Thermal::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Generates a the random thermal field of a *SpinSystem*");
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
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
	
	if(func == l_apply)
	{
		lua_pushstring(L, "Generates a the random thermal field of a *SpinSystem*");
		lua_pushstring(L, "1 *Random*, 1 *SpinSystem*: The first argument is a random number generator that is used as a source of random values. The second argument is the spin system which will receive the field.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_scalesite)
	{
		lua_pushstring(L, "Scale the thermal field at a site. This allows non-uniform thermal effects over a lattice.");
		lua_pushstring(L, "1 *3Vector*, 1 Number: The vectors define the lattice sites that will have a scaled thermal effect, the number is the how the thermal field is scaled.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_settemp)
	{
		lua_pushstring(L, "Sets the base value of the temperature. ");
		lua_pushstring(L, "1 number: temperature of the system.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_gettemp)
	{
		lua_pushstring(L, "Gets the base value of the temperature. ");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 number: temperature of the system.");
		return 3;
	}

	return SpinOperation::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Thermal::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, SpinOperation::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"apply",        l_apply},
		{"scaleSite",    l_scalesite},
		{"setTemperature", l_settemp},
		{"set",          l_settemp},
		{"get",          l_gettemp},
		{"temperature",  l_gettemp},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}



#include "info.h"
extern "C"
{
THERMAL_API int lib_register(lua_State* L);
THERMAL_API int lib_version(lua_State* L);
THERMAL_API const char* lib_name(lua_State* L);
THERMAL_API int lib_main(lua_State* L);
}

THERMAL_API int lib_register(lua_State* L)
{
	luaT_register<Thermal>(L);
	return 0;
}

THERMAL_API int lib_version(lua_State* L)
{
	return __revi;
}

THERMAL_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Thermal";
#else
	return "Thermal-Debug";
#endif
}

THERMAL_API int lib_main(lua_State* L)
{
	return 0;
}


