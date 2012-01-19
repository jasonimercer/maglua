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
#include <math.h>
#include <stdio.h>

#ifndef WIN32
#include <time.h>
#endif

#include "spinsystem.hpp"
#include "spinoperationthermal.hpp"

Thermal::Thermal(int nx, int ny, int nz)
	: SpinOperation("Thermal", THERMAL_SLOT, nx, ny, nz, ENCODE_THERMAL)
{
	d_scale = 0;
	temperature = 0;
 	registerWS();

	init();
}

void Thermal::sync_dh(bool force)
{
	if(new_device || force)
	{
		ss_copyDeviceToHost(h_scale, d_scale, nxyz);
		
		new_host = false;
		new_device = false;
	}	
}

void Thermal::sync_hd(bool force)
{
	if(new_host || force)
	{
		ss_copyHostToDevice(d_scale, h_scale, nxyz);

		new_host = false;
		new_device = false;
	}	
}


void Thermal::encode(buffer* b)
{
	sync_dh();
 	encodeInteger(nx, b);
 	encodeInteger(ny, b);
 	encodeInteger(nz, b);
 	
 	encodeDouble(temperature, b);
 	
 	for(int i=0; i<nxyz; i++)
 		encodeDouble(h_scale[i], b);
}

int  Thermal::decode(buffer* b)
{
	deinit();
	
	nx = decodeInteger(b);
	ny = decodeInteger(b);
	nz = decodeInteger(b);
	nxyz = nx * ny * nz;

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

	malloc_device(&d_scale, sizeof(double)*nx*ny*nz);
	malloc_host  (&h_scale, sizeof(double)*nx*ny*nz);
// 	ss_d_make3DArray(, nx, ny, nz);
// 	ss_h_make3DArray(&h_scale, nx, ny, nz);
	
	for(int i=0; i<nx*ny*nz; i++)
	{
		h_scale[i] = 1;
	}
	new_host = true;
	sync_hd();
	
	twiddle = 1;
	HybridTausAllocState(&d_state, nx, ny, nz);
	HybridTausAllocRNG(&d_rngs, nx, ny, nz);
	
#ifndef WIN32
#warning Random seed by time(). Need something more elegant
	HybridTausSeed(d_state, nx, ny, nz, time(0));
#else
	HybridTausSeed(d_state, nx, ny, nz, 123456);
#endif
}

void Thermal::deinit()
{
	if(d_scale == 0)
		return;

	free_device(d_scale);
	free_host  (h_scale);
// 	ss_d_free3DArray(d_scale);
// 	ss_h_free3DArray(h_scale);
	
	HybridTausFreeState(d_state);
	HybridTausFreeRNG(d_rngs);
	
	d_scale = 0;

}
	
Thermal::~Thermal()
{
	unregisterWS();
	deinit();
}

bool Thermal::applyToSum(RNG* , SpinSystem* ss)
{
	sync_hd();
	ss->sync_spins_hd();
	ss->ensureSlotExists(slot);

	if(twiddle)
	{
		HybridTaus_get6Normals(d_state, d_rngs, nx, ny, nz);
		twiddle = 0;
	}
	else
		twiddle = 1;
	
// 	markSlotUsed(ss);
	const int sz = sizeof(double)*nxyz;
	double* d_wsx;
	double* d_wsy;
	double* d_wsz;
	
	getWSMem(&d_wsx, sz,
			 &d_wsy, sz,
			 &d_wsz, sz);
	
//     double* d_wsAll = (double*)getWSMem(sizeof(double)*nxyz*3);
//     double* d_wsx = d_wsAll + nxyz * 0;
//     double* d_wsy = d_wsAll + nxyz * 1;
//     double* d_wsz = d_wsAll + nxyz * 2;

	const double alpha = ss->alpha;
	const double dt    = ss->dt;
	const double gamma = ss->gamma;
	
	double* d_hx = ss->d_hx[slot];
	double* d_hy = ss->d_hy[slot];
	double* d_hz = ss->d_hz[slot];

	cuda_thermal(d_rngs, twiddle, 
		alpha, gamma, dt, temperature,
		d_wsx, d_wsy, d_wsz, ss->d_ms,
		d_scale,
		nx, ny, nz);

	
	const int nxyz = nx*ny*nz;
	cuda_addArrays(ss->d_hx[SUM_SLOT], nxyz, ss->d_hx[SUM_SLOT], d_wsx);
	cuda_addArrays(ss->d_hy[SUM_SLOT], nxyz, ss->d_hy[SUM_SLOT], d_wsy);
	cuda_addArrays(ss->d_hz[SUM_SLOT], nxyz, ss->d_hz[SUM_SLOT], d_wsz);
// 	ss->new_device_fields[slot] = true;
	ss->slot_used[SUM_SLOT] = true;

	return true;
}


bool Thermal::apply(RNG* , SpinSystem* ss)
{
	sync_hd();
	ss->sync_spins_hd();
	markSlotUsed(ss);
	ss->ensureSlotExists(slot);

	if(twiddle)
	{
		HybridTaus_get6Normals(d_state, d_rngs, nx, ny, nz);
		twiddle = 0;
	}
	else
		twiddle = 1;
	

	const double alpha = ss->alpha;
	const double dt    = ss->dt;
	const double gamma = ss->gamma;
	
	double* d_hx = ss->d_hx[slot];
	double* d_hy = ss->d_hy[slot];
	double* d_hz = ss->d_hz[slot];

	cuda_thermal(d_rngs, twiddle, 
		alpha, gamma, dt, temperature,
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



Thermal* checkThermal(lua_State* L, int idx)
{
	Thermal** pp = (Thermal**)luaL_checkudata(L, idx, "MERCER.thermal");
    luaL_argcheck(L, pp != NULL, 1, "`Thermal' expected");
    return *pp;
}

void lua_pushThermal(lua_State* L, Encodable* _th)
{
	Thermal* th = dynamic_cast<Thermal*>(_th);
	if(!th) return;
	
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
// 	RNG*       rng = checkRandom(L, 2);
	SpinSystem* ss = checkSpinSystem(L, 3);
// 	SpinSystem* ss = checkSpinSystem(L, 2);

	if(!th)
		return luaL_error(L, "Thermal object required");

// 	if(!rng || !ss || !llg)
// 		return luaL_error(L, "Thermal.apply requires llg, rng, spinsystem");
	
	if(!ss)
		return luaL_error(L, "Thermal.apply requires spinsystem");
	
	if(!th->apply(0,ss))
		return luaL_error(L, th->errormsg.c_str());
	
	return 0;
}

int l_thermal_applytosum(lua_State* L)
{
	Thermal*    th = checkThermal(L, 1);
// 	RNG*       rng = checkRandom(L, 2);
	SpinSystem* ss = checkSpinSystem(L, 3);
// 	SpinSystem* ss = checkSpinSystem(L, 2);

	if(!th)
		return luaL_error(L, "Thermal object required");

// 	if(!rng || !ss || !llg)
// 		return luaL_error(L, "Thermal.apply requires llg, rng, spinsystem");
	
	if(!ss)
		return luaL_error(L, "Thermal.apply requires spinsystem");
	
	if(!th->applyToSum(0,ss))
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
		
	if(func == l_thermal_applytosum)
	{
		lua_pushstring(L, "Generates a the random thermal field of a *SpinSystem*");
		lua_pushstring(L, "1 *Random*, 1 *SpinSystem*: The first argument is a random number generator that is used as a source of random values. The second argument is the spin system which will receive the field, added to the total field.");
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

static Encodable* newThing()
{
	return new Thermal;
}

void registerThermal(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_thermal_gc},
		{"__tostring",   l_thermal_tostring},
		{"apply",        l_thermal_apply},
		{"applyToSum",   l_thermal_applytosum},
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
	Factory_registerItem(ENCODE_THERMAL, newThing, lua_pushThermal, "Thermal");
}

#include "info.h"
extern "C"
{
THERMALCUDA_API int lib_register(lua_State* L);
THERMALCUDA_API int lib_version(lua_State* L);
THERMALCUDA_API const char* lib_name(lua_State* L);
THERMALCUDA_API int lib_main(lua_State* L, int argc, char** argv);
}

THERMALCUDA_API int lib_register(lua_State* L)
{
	registerThermal(L);
	return 0;
}

THERMALCUDA_API int lib_version(lua_State* L)
{
	return __revi;
}

THERMALCUDA_API const char* lib_name(lua_State* L)
{
	return "Thermal-Cuda";
}

THERMALCUDA_API int lib_main(lua_State* L, int argc, char** argv)
{
	return 0;
}


