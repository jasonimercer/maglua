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

#define FALSE 0
#define TRUE 1

RNG::RNG()
	: LuaBaseObject(hash32(lineage(0)))
{
	__gaussStep = 1;
}

int RNG::luaInit(lua_State* L)
{
	if(lua_isnumber(L, 1))
	{
		int s = lua_tointeger(L, 1);
		if(s)
		{
			seed(s);
			return 0;
		}
	}
	seed();
	return LuaBaseObject::luaInit(L);
}

void RNG::seed()
{
    // First try getting an array from /dev/urandom
#ifndef WIN32
	long _seed;
    FILE* urandom = fopen( "/dev/urandom", "rb" );
    if( urandom )
    {
        if(fread(&_seed, sizeof(_seed), 1, urandom))
        {
            fclose(urandom);
			seed(_seed);
            return;
        }
        fclose(urandom);
    }
#endif
    seed( time(0) );
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





static int l_seed(lua_State* L)
{
	LUA_PREAMBLE(RNG, r, 1);	
	r->seed(lua_tointeger(L, 2));
	return 0;
}

static int l_uniform(lua_State* L)
{
	LUA_PREAMBLE(RNG, r, 1);	
	if(lua_isnumber(L, 2))
		lua_pushnumber(L, r->rand(lua_tonumber(L, 2)));
	else
		lua_pushnumber(L, r->rand());
	
	return 1;
}

static int l_normal(lua_State* L)
{
	LUA_PREAMBLE(RNG, r, 1);	
	lua_pushnumber(L, r->randNorm());
	return 1;
}


int RNG::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Random number generator abstract base class");
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
	

	if(func == l_seed)
	{
		lua_pushstring(L, "Set the seed for the random number generator");
		lua_pushstring(L, "1 integer: This will be used as the seed for the RNG.");
		lua_pushstring(L, "");
		return 3;
	}
		
	if(func == l_uniform)
	{
		lua_pushstring(L, "Return a value selected from a uniform distribution");
		lua_pushstring(L, "1 Optional number:");
		lua_pushstring(L, "1 number: A random uniform number in the range of [0:1] if for no optional value or [0:n] for an optional argument n");
		return 3;
	}

	if(func == l_normal)
	{
		lua_pushstring(L, "Return a value selected from a normal distribution");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 number: A random normal number selected for a normal distribution with stddev 1 and mean 0.");
		return 3;
	}
	
	return 0;
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* RNG::luaMethods()
{
	if(m[127].name)
		return m;

	static const luaL_Reg _m[] =
	{
		{"setSeed",      l_seed},
		{"uniform",      l_uniform},
		{"normal",       l_normal},
		{"rand",         l_uniform},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}



#include "info.h"
extern "C"
{
RANDOM_API int lib_register(lua_State* L);
RANDOM_API int lib_version(lua_State* L);
RANDOM_API const char* lib_name(lua_State* L);
RANDOM_API int lib_main(lua_State* L);
}

#include "random_wrapper.h"
RANDOM_API int lib_register(lua_State* L)
{
	luaT_register<RNG>(L);
	luaT_register<CRand>(L);
	luaT_register<MTRand>(L);
	luaT_register<Isaac>(L);

        luaL_dofile_random_wrapper(L);

	return 0;
}

RANDOM_API int lib_version(lua_State* L)
{
	return __revi;
}

RANDOM_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Random";
#else
	return "Random-Debug";
#endif
}

RANDOM_API int lib_main(lua_State* L)
{
	return 0;
}
