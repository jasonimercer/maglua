/******************************************************************************
* Copyright (C) 2008-2012 Jason Mercer.  All rights reserved.
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
#include <math.h>
#include <stdio.h>

#ifndef WIN32
#include <time.h>
#endif

#include "spinsystem.hpp"
#include "spinoperationthermal.hpp"

Thermal::Thermal(int nx, int ny, int nz)
	: SpinOperation(Thermal::typeName(), THERMAL_SLOT, nx, ny, nz, hash32(Thermal::typeName()))
{
	scale = 0;
	temperature = 0;
 	registerWS();

	init();
}

int Thermal::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	init();
	return 0;
}

void Thermal::encode(buffer* b)
{
	SpinOperation::encode(b);

 	encodeDouble(temperature, b);
	scale->encode(b);
}

int  Thermal::decode(buffer* b)
{
	deinit();
	SpinOperation::decode(b);
	init();
	
	temperature = decodeDouble(b);
	scale->decode(b);
	return 0;
}

void Thermal::init()
{
	if(scale != 0)
		deinit();

	scale = luaT_inc<dArray>(new dArray(nx, ny, nz));
	scale->setAll(1.0);
}

void Thermal::deinit()
{
	luaT_dec<dArray>(scale);
	scale = 0;
}
	
Thermal::~Thermal()
{
	unregisterWS();
	deinit();
}

#include "../random_cuda/hybridtaus.h"
bool Thermal::applyToSum(RNG* rng, SpinSystem* ss)
{
	HybridTaus* ht = dynamic_cast<HybridTaus*>(rng);

	if(!ht)
	{
		errormsg = "CUDA Thermal calculations require a GPU based Random Number Generator (HybrisTaus)";
		return false;
	}

	ss->ensureSlotExists(slot);

	int twiddle = 0;
	float* d_rngs = ht->get6Normals(nx,ny,nz,twiddle);

	const int sz = sizeof(double)*nxyz;
	double* d_wsx;
	double* d_wsy;
	double* d_wsz;
	
	getWSMem3(&d_wsx, sz,
			  &d_wsy, sz,
			  &d_wsz, sz);

	const double alpha = ss->alpha;
	const double dt    = ss->dt;
	const double gamma = ss->gamma;
	
	cuda_thermal(d_rngs, twiddle, 
		alpha, gamma, dt, temperature * global_scale,
		d_wsx, d_wsy, d_wsz, ss->ms->ddata(),
		scale->ddata(),
		nx*ny*nz);

	const int nxyz = nx*ny*nz;
	arraySumAll(ss->hx[SUM_SLOT]->ddata(), ss->hx[SUM_SLOT]->ddata(), d_wsx, nxyz);
	arraySumAll(ss->hy[SUM_SLOT]->ddata(), ss->hy[SUM_SLOT]->ddata(), d_wsy, nxyz);
	arraySumAll(ss->hz[SUM_SLOT]->ddata(), ss->hz[SUM_SLOT]->ddata(), d_wsz, nxyz);
	
	ss->hx[SUM_SLOT]->new_device = true;
	ss->hy[SUM_SLOT]->new_device = true;
	ss->hz[SUM_SLOT]->new_device = true;
	ss->slot_used[SUM_SLOT] = true;
	
	
	return true;
}


bool Thermal::apply(RNG* rng, SpinSystem* ss)
{
	HybridTaus* ht = dynamic_cast<HybridTaus*>(rng);

	if(!ht)
	{
		errormsg = "CUDA Thermal calculations require a GPU based Random Number Generator (HybridTaus)";
		return false;
	}

	ss->ensureSlotExists(slot);
	markSlotUsed(ss);

	//this twiddle is output in get6Normals
	int twiddle = 0;
	float* d_rngs = ht->get6Normals(nx,ny,nz,twiddle);

	const double alpha = ss->alpha;
	const double dt    = ss->dt;
	const double gamma = ss->gamma;
	
	double* d_hx = ss->hx[slot]->ddata();
	double* d_hy = ss->hy[slot]->ddata();
	double* d_hz = ss->hz[slot]->ddata();

	cuda_thermal(d_rngs, twiddle, 
		alpha, gamma, dt, temperature * global_scale,
		d_hx, d_hy, d_hz, ss->ms->ddata(),
		scale->ddata(),
		nx*ny*nz);
	
	ss->hx[slot]->new_device = true;
	ss->hy[slot]->new_device = true;
	ss->hz[slot]->new_device = true;
	return true;
}


void Thermal::scaleSite(int px, int py, int pz, double strength)
{
	if(member(px, py, pz))
	{
		int idx = getidx(px, py, pz);
		(*scale)[idx] = strength;
	}
}



static int l_apply(lua_State* L)
{
	LUA_PREAMBLE(Thermal, th, 1);
	LUA_PREAMBLE(RNG, rng, 2);
	LUA_PREAMBLE(SpinSystem, ss, 3);

	if(!th->apply(rng,ss))
		return luaL_error(L, th->errormsg.c_str());
	
	return 0;
}

static int l_applytosum(lua_State* L)
{
	LUA_PREAMBLE(Thermal, th, 1);
	LUA_PREAMBLE(RNG, rng, 2);
	LUA_PREAMBLE(SpinSystem, ss, 3);

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

static int l_getscalearray(lua_State* L)
{
	LUA_PREAMBLE(Thermal, th, 1);
	luaT_push<dArray>(L, th->scale);
	return 1;
}
static int l_setscalearray(lua_State* L)
{
	LUA_PREAMBLE(Thermal, th, 1);
	LUA_PREAMBLE(dArray, a, 2);
	if(a->sameSize(th->scale))
	{
		luaT_inc<dArray>(a);
		luaT_dec<dArray>(th->scale);
		th->scale = a;
		return 0;
	}
	return luaL_error(L, "Array size mismatch\n");
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
		lua_pushstring(L, "1 *Random*, 1 *SpinSystem*: The first argument is a random number generator that is used as a source of random values. The second argument is the spin system which will receive the field. Note: The RNG must be GPU based.");
		lua_pushstring(L, "");
		return 3;
	}	
	if(func == l_applytosum)
	{
		lua_pushstring(L, "Generates a the random thermal field of a *SpinSystem*");
		lua_pushstring(L, "1 *Random*, 1 *SpinSystem*: The first argument is a random number generator that is used as a source of random values. The second argument is the spin system which will receive the field summed to the previous value. Note: The RNG must be GPU based.");
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
	
	if(func == l_getscalearray)
	{
		lua_pushstring(L, "Get array object reprenenting thermal scaling for each site. ");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Array");
		return 3;
	}
	if(func == l_setscalearray)
	{
		lua_pushstring(L, "Set array object reprenenting thermal scaling for each site. ");
		lua_pushstring(L, "1 Array");
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

const luaL_Reg* Thermal::luaMethods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)return m;

	merge_luaL_Reg(m, SpinOperation::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"apply",        l_apply},
		{"applyToSum",        l_applytosum},
		{"scaleSite",    l_scalesite},
		{"setTemperature", l_settemp},
		{"set",          l_settemp},
		{"get",          l_gettemp},
		{"temperature",  l_gettemp},
		
		{"scaleArray",  l_getscalearray},
		{"setScaleArray",  l_setscalearray},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}



#include "info.h"
extern "C"
{
THERMALCUDA_API int lib_register(lua_State* L);
THERMALCUDA_API int lib_version(lua_State* L);
THERMALCUDA_API const char* lib_name(lua_State* L);
THERMALCUDA_API int lib_main(lua_State* L);
}

int lib_register(lua_State* L)
{
	luaT_register<Thermal>(L);
	return 0;
}

int lib_version(lua_State* L)
{
	return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Thermal-Cuda";
#else
	return "Thermal-Cuda-Debug";
#endif
}

int lib_main(lua_State* L)
{
	return 0;
}


