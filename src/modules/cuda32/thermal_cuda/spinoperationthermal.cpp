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
	d_scale = 0;
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

void Thermal::sync_dh(bool force)
{
	if(new_device || force)
	{
		ss_copyDeviceToHost32(h_scale, d_scale, nxyz);
		
		new_host = false;
		new_device = false;
	}	
}

void Thermal::sync_hd(bool force)
{
	if(new_host || force)
	{
		ss_copyHostToDevice32(d_scale, h_scale, nxyz);

		new_host = false;
		new_device = false;
	}	
}


void Thermal::encode(buffer* b)
{
	sync_dh();
	SpinOperation::encode(b);

 	encodeDouble(temperature, b);
 	
 	for(int i=0; i<nxyz; i++)
 		encodeDouble(h_scale[i], b);
}

int  Thermal::decode(buffer* b)
{
	deinit();
	SpinOperation::decode(b);
	init();
	
	temperature = decodeDouble(b);
	
	for(int i=0; i<nxyz; i++)
		h_scale[i] = decodeDouble(b);
	
	new_host = true;
	new_device = false;
	return 0;
}

void Thermal::init()
{
	if(d_scale != 0)
		deinit();

	malloc_device(&d_scale, sizeof(float)*nx*ny*nz);
	malloc_host  (&h_scale, sizeof(float)*nx*ny*nz);
	for(int i=0; i<nx*ny*nz; i++)
	{
		h_scale[i] = 1;
	}
	new_host = true;
	sync_hd();
}

void Thermal::deinit()
{
	if(d_scale == 0)
		return;

	free_device(d_scale);
	free_host  (h_scale);
	
	d_scale = 0;

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

	sync_hd();
	ss->sync_spins_hd();
	ss->ensureSlotExists(slot);

	int twiddle = 0;
	float* d_rngs = ht->get6Normals(nx,ny,nz,twiddle);

	const int sz = sizeof(float)*nxyz;
	float* d_wsx;
	float* d_wsy;
	float* d_wsz;
	
	getWSMem(&d_wsx, sz,
			 &d_wsy, sz,
			 &d_wsz, sz);

	const float alpha = ss->alpha;
	const float dt    = ss->dt;
	const float gamma = ss->gamma;
	
	cuda_thermal32(d_rngs, twiddle, 
		alpha, gamma, dt, temperature * global_scale,
		d_wsx, d_wsy, d_wsz, ss->d_ms,
		d_scale,
		nx, ny, nz);
	
	const int nxyz = nx*ny*nz;
	cuda_addArrays32(ss->d_hx[SUM_SLOT], nxyz, ss->d_hx[SUM_SLOT], d_wsx);
	cuda_addArrays32(ss->d_hy[SUM_SLOT], nxyz, ss->d_hy[SUM_SLOT], d_wsy);
	cuda_addArrays32(ss->d_hz[SUM_SLOT], nxyz, ss->d_hz[SUM_SLOT], d_wsz);
	ss->slot_used[SUM_SLOT] = true;

	return true;
}


bool Thermal::apply(RNG* rng, SpinSystem* ss)
{
	HybridTaus* ht = dynamic_cast<HybridTaus*>(rng);

	if(!ht)
	{
		errormsg = "CUDA Thermal calculations require a GPU based Random Number Generator (HybrisTaus)";
		return false;
	}

	sync_hd();
	ss->sync_spins_hd();
	ss->ensureSlotExists(slot);

	int twiddle = 0;
	float* d_rngs = ht->get6Normals(nx,ny,nz,twiddle);

	const float alpha = ss->alpha;
	const float dt    = ss->dt;
	const float gamma = ss->gamma;
	
	float* d_hx = ss->d_hx[slot];
	float* d_hy = ss->d_hy[slot];
	float* d_hz = ss->d_hz[slot];

	cuda_thermal32(d_rngs, twiddle, 
		alpha, gamma, dt, temperature * global_scale,
		d_hx, d_hy, d_hz, ss->d_ms,
		d_scale,
		nx, ny, nz);
	
	ss->new_device_fields[slot] = true;

	return true;
}


void Thermal::scaleSite(int px, int py, int pz, double strength)
{
	if(member(px, py, pz))
	{
		sync_dh();
		int idx = getidx(px, py, pz);
		h_scale[idx] = strength;
		new_host = true;
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
	return "Thermal-Cuda32";
#else
	return "Thermal-Cuda32-Debug";
#endif
}

int lib_main(lua_State* L)
{
	return 0;
}


