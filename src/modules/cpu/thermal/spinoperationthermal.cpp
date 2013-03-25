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
	scale = luaT_inc<dArray>(new dArray(nx,ny,nz));
	scale->setAll(1.0);
	temperature = 0;
	
	myRNG = 0;
}

int Thermal::luaInit(lua_State* L)
{
	luaT_dec<dArray>(scale);

	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz

	scale = luaT_inc<dArray>(new dArray(nx,ny,nz));
	scale->setAll(1.0);
	temperature = 0;
	
	if(luaT_is<RNG>(L, -1))
	{
		myRNG = luaT_to<RNG>(L, -1);
		luaT_inc<RNG>(myRNG);
	}
	else
		myRNG = 0;
	
	return 0;
}

void Thermal::encode(buffer* b)
{
	SpinOperation::encode(b);
	char version = 0;
	encodeChar(version, b);
	
	encodeDouble(temperature, b);
	
	scale->encode(b);
}

int  Thermal::decode(buffer* b)
{
	SpinOperation::decode(b);
	char version = decodeChar(b);
	if(version == 0)
	{
		temperature = decodeDouble(b);
		
		luaT_dec<dArray>(scale);
		scale = luaT_inc<dArray>(new dArray(nx,ny,nz));
		scale->decode(b);
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}
	return 0;
}

Thermal::~Thermal()
{
	luaT_dec<dArray>(scale);
	luaT_dec<RNG>(myRNG);
}


class gamma_alpha
{
public:
	gamma_alpha(SpinSystem* ss)
	{
		// making local refcounted references to data so the
		// arrays don't get free'd in mid-operation (from another thread)
		site_alpha = luaT_inc<dArray>(ss->site_alpha);
		site_gamma = luaT_inc<dArray>(ss->site_gamma);
		
		a_alpha = (site_alpha)?ss->site_alpha->data():0;
		a_gamma = (site_gamma)?ss->site_gamma->data():0;
		
		v_alpha = ss->alpha;
		v_gamma = ss->gamma;
	}
	
	~gamma_alpha()
	{
		luaT_dec<dArray>(site_alpha);
		luaT_dec<dArray>(site_gamma);
	}
	
	double gamma(int idx)
	{
		if(a_gamma)
			return a_gamma[idx];
		return v_gamma;
	}
	double alpha(int idx)
	{
		if(a_alpha)
			return a_alpha[idx];
		return v_alpha;
	}
	
	dArray* site_alpha;
	dArray* site_gamma;
	
	double* a_gamma;
	double* a_alpha;
	
	double v_gamma;
	double v_alpha;
};

bool Thermal::apply(SpinSystem* ss, RNG* useThisRNG)
{
	markSlotUsed(ss);
	
	RNG* rand = myRNG;
	if(useThisRNG)
		rand = useThisRNG;

	if(rand == 0)
	{
		errormsg = "Missing RNG";
		return false;
	}
	
	const double alpha = ss->alpha;
	const double dt    = ss->dt;
	const double gamma = ss->gamma;

	double* hx = ss->hx[slot]->data();
	double* hy = ss->hy[slot]->data();
	double* hz = ss->hz[slot]->data();
	const double* ms = ss->ms->data();

	
	gamma_alpha ga(ss);
	
	for(int i=0; i<ss->nxyz; i++)
	{
		if(ms[i] != 0 && temperature != 0)
		{
			const double alpha = ga.alpha(i);
			const double gamma = ga.gamma(i);
			if(gamma > 0)
			{
				const double stddev = sqrt((2.0 * alpha * global_scale * temperature * (*scale)[i]) / (ms[i] * dt * gamma));
				
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
	int idx;

	if(scale->member(px,py,pz, idx))
	{
		(*scale)[idx] = strength;
	}
}



static int l_apply(lua_State* L)
{
	LUA_PREAMBLE(Thermal, th, 1);
	LUA_PREAMBLE(SpinSystem,ss,2);

	if(luaT_is<RNG>(L, 3))
	{
		if(!th->apply(ss, luaT_to<RNG>(L, 3)))
			return luaL_error(L, th->errormsg.c_str());
	}
	if(!th->apply(ss, 0))
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
	LUA_PREAMBLE(dArray, s, 1);
	if(th->scale->sameSize(s))
	{
		luaT_inc<dArray>(s);
		luaT_dec<dArray>(th->scale);
		th->scale = s;
	}
	else
	{
		return luaL_error(L, "Array size mismatch");
	}
	return 0;
}

static int l_rng(lua_State* L)
{
	LUA_PREAMBLE(Thermal, th, 1);

	luaT_push<RNG>(L, th->myRNG);
	return 1;
}

int Thermal::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Generates a the random thermal field of a *SpinSystem*");
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*, 1 Optional *Random*: System Size and built in RNG"); 
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
		lua_pushstring(L, "1 *SpinSystem*, 1 Optional *Random*,: The first argument is the spin system which will receive the field. The second optional argument is a random number generator that is used as a source of random values, if absent then the RNG supplied in the constructor will be used.");
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
	
	if(func == l_getscalearray)
	{
		lua_pushstring(L, "Get an array representing the thermal scale at each site. This array is connected to the Operator so changes to the returned array will change the Operator.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Array: The thermal scale of the sites.");
		return 3;
	}
	if(func == l_setscalearray)
	{
		lua_pushstring(L, "Set an array representing the new thermal scale at each site.");
		lua_pushstring(L, "1 Array: The thermal scale of the sites.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_rng)
	{
		lua_pushstring(L, "Get the *Random* number generator supplied at initialization");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 *Random* or 1 nil: RNG");
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
		{"arrayScale",  l_getscalearray},
		{"setArrayScale",  l_setscalearray},
		{"random",        l_rng},
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


