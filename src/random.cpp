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
		{NULL, NULL}
	};
	
	luaL_register(L, "Random", functions);
	lua_pop(L,1);	
}
