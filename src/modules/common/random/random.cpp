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

#include "random.h"

#include <string.h>
#include "isaac.h"
#include "mersennetwister.h"
#include "crand.h"

RNG::RNG(const char* name)
{
	type = name;
	__gaussStep = 1;
}

RNG* checkRandom(lua_State* L, int idx)
{
	RNG** pp = (RNG**)luaL_checkudata(L, idx, "MERCER.rng");
    luaL_argcheck(L, pp != NULL, 1, "`RNG' expected");
    return *pp;
}

double RNG::rand()                        // real number in [0,1]
{
	return double(randInt()) * (1.0/4294967295.0);
}

double RNG::rand( const double n )        // real number in [0,n]
{
	return rand() * n;
}

double RNG::randExc()                     // real number in [0,1)
{
	return double(randInt()) * (1.0/4294967296.0);
}

double RNG::randExc( const double n )     // real number in [0,n)
{
	return randExc() * n;
}

double RNG::randDblExc()                  // real number in (0,1)
{
	return ( double(randInt()) + 0.5 ) * (1.0/4294967296.0);
}

double RNG::randDblExc( const double n )  // real number in (0,n)
{
	return randDblExc() * n;
}



// Access to nonuniform random number distributions
double RNG::randNorm( const double mean, const double stddev )
{
	if(__gaussStep)
	{
		double v1, v2, t;

		do{
			v1 = rand()*2.0-1.0;
			v2 = rand()*2.0-1.0;
			t = v1*v1+v2*v2;
		}while(t >= 1.0);
		

		double sq= sqrt(-2.0 * log(t) / t);
				
		gaussPair[0] = sq * v1;
		gaussPair[1] = sq * v2;

		__gaussStep = FALSE;
		return mean + gaussPair[0] * stddev;
	}
	__gaussStep = TRUE;
	return mean + gaussPair[1] * stddev;

}




int l_rand_new(lua_State* L)
{
	RNG* r = 0;
	const char* _error = "First argument of RNG.new must be `MersenneTwister', `Isaac' or `CRandom'";
	if(!lua_isstring(L, 1))
		return luaL_error(L, _error);
	
	const char* type = lua_tostring(L, 1);
	
	if(strcasecmp(type, "mersennetwister") == 0)
	{
		r = new MTRand;
	}
	else if(strcasecmp(type, "isaac") == 0)
	{
		r = new Isaac;
	}
	else if(strcasecmp(type, "crandom") == 0)
	{
		r = new CRand;
	}
	
	if(!r)
		return luaL_error(L, _error);

	if(lua_isnumber(L, 2))
		r->seed(lua_tointeger(L, 2));
	else
		r->seed();
	
	RNG** pp = (RNG**)lua_newuserdata(L, sizeof(RNG**));
	
	*pp = r;
	luaL_getmetatable(L, "MERCER.rng");
	lua_setmetatable(L, -2);
	return 1;
}


int l_rand_gc(lua_State* L)
{
	RNG* r = checkRandom(L, 1);
	delete r;
	
	return 0;
}

int l_rand_seed(lua_State* L)
{
	RNG* r = checkRandom(L, 1);
	if(!r) return 0; 
	
	r->seed(lua_tointeger(L, 2));

	return 0;
}

int l_rand_uniform(lua_State* L)
{
	RNG* r = checkRandom(L, 1);
	if(!r) return 0; 
	
	if(lua_isnumber(L, 2))
		lua_pushnumber(L, r->rand(lua_tonumber(L, 2)));
	else
		lua_pushnumber(L, r->rand());
	
	return 1;
}

int l_rand_normal(lua_State* L)
{
	RNG* r = checkRandom(L, 1);
	if(!r) return 0; 
	
	lua_pushnumber(L, r->randNorm());
	return 1;
}


static int l_rand_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.rng");
	return 1;
}

static int l_rand_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Random creates a random number generator using one of numerous algorithms.");
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
	
	if(func == l_rand_new)
	{
		lua_pushstring(L, "Create a new Random object.");
		lua_pushstring(L, "1 string: The string argument defines the Random type. It may be one of the following:\n\"MersenneTwister\" - use the Mersenne Twister RNG by Makoto Matsumoto.\n\"Isaac\" - ISAAC RNG by Robert Jenkins.\n\"CRandom\" - built in C random number (rand_r). CRandom does not suffer poor randomization on lower order bits as in older rand() implementations."); 
		lua_pushstring(L, "1 Random object");
		return 3;
	}
	
	if(func == l_rand_seed)
	{
		lua_pushstring(L, "Set the seed for the random number generator");
		lua_pushstring(L, "1 integer: This will be used as the seed for the RNG.");
		lua_pushstring(L, "");
		return 3;
	}
		
	if(func == l_rand_uniform)
	{
		lua_pushstring(L, "Return a value selected from a uniform distribution");
		lua_pushstring(L, "1 Optional number:");
		lua_pushstring(L, "1 number: A random uniform number in the range of [0:1] if for no optional value or [0:n] for an optional argument n");
		return 3;
	}

	if(func == l_rand_normal)
	{
		lua_pushstring(L, "Return a value selected from a normal distribution");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 number: A random normal number selected for a normal distribution with stddev 1 and mean 0.");
		return 3;
	}
	
	return 0;
}


// static Encodable* newMTRand()
// {
// 	return new MTRand;
// }
// static Encodable* newIsaac()
// {
// 	return new Isaac;
// }
// static Encodable* newCRand()
// {
// 	return new CRand;
// }

void registerRandom(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_rand_gc},
		{"setSeed",      l_rand_seed},
		{"uniform",      l_rand_uniform},
		{"normal",       l_rand_normal},
		{"rand",         l_rand_uniform},
		{NULL, NULL}
	};
	
	luaL_newmetatable(L, "MERCER.rng");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
	
	static const struct luaL_reg functions [] = {
		{"new",                 l_rand_new},
		{"help",                l_rand_help},
		{"metatable",           l_rand_mt},
		{NULL, NULL}
	};
	
	luaL_register(L, "Random", functions);
	lua_pop(L,1);
}



#include "info.h"
extern "C"
{
RANDOM_API int lib_register(lua_State* L);
RANDOM_API int lib_version(lua_State* L);
RANDOM_API const char* lib_name(lua_State* L);
RANDOM_API int lib_main(lua_State* L);
}

RANDOM_API int lib_register(lua_State* L)
{
	registerRandom(L);
	return 0;
}

RANDOM_API int lib_version(lua_State* L)
{
	return __revi;
}

RANDOM_API const char* lib_name(lua_State* L)
{
#ifdef NDEBUG 
	return "Random";
#else
	return "Random-Debug";
#endif
}

RANDOM_API int lib_main(lua_State* L)
{
	return 0;
}
