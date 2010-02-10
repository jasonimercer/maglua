#include "spinoperationexchange.h"
#include "spinsystem.h"

#include <stdlib.h>
#include <stdio.h>

Exchange::Exchange(int nx, int ny, int nz)
	: SpinOperation("Exchange", EXCHANGE_SLOT, nx, ny, nz)
{
	size = 32;
	num  = 0;

	fromsite =    (int*)malloc(sizeof(int) * size);
	  tosite =    (int*)malloc(sizeof(int) * size);
	strength = (double*)malloc(sizeof(double) * size);
}

Exchange::~Exchange()
{
	free(fromsite);
	free(  tosite);
	free(strength);
}

bool Exchange::apply(SpinSystem* ss)
{
	double* hx = ss->hx[slot];
	double* hy = ss->hy[slot];
	double* hz = ss->hz[slot];

	const double* sx = ss->x;
	const double* sy = ss->y;
	const double* sz = ss->z;

	for(int i=0; i<num; i++)
	{
		const int t    =   tosite[i];
		const int f    = fromsite[i];
		const double s = strength[i];
		
		hx[t] += ss->x[f] * s;
		hy[t] += ss->y[f] * s;
		hz[t] += ss->z[f] * s;
	}
	return true;
}

void Exchange::addPath(int site1, int site2, double str)
{
	if(num + 1 == size)
	{
		size *= 2;
		fromsite =    (int*)realloc(fromsite, sizeof(int) * size);
		  tosite =    (int*)realloc(  tosite, sizeof(int) * size);
		strength = (double*)realloc(strength, sizeof(double) * size);
	}
	
	fromsite[num] = site1;
	  tosite[num] = site2;
	strength[num] = str;
	num++;
}



Exchange* checkExchange(lua_State* L, int idx)
{
	Exchange** pp = (Exchange**)luaL_checkudata(L, idx, "MERCER.exchange");
    luaL_argcheck(L, pp != NULL, 1, "`Exchange' expected");
    return *pp;
}


int l_ex_new(lua_State* L)
{
	if(lua_gettop(L) != 3)
		return luaL_error(L, "Exchange.new requires nx, ny, nz");

	Exchange* ex = new Exchange(
			lua_tointeger(L, 1),
			lua_tointeger(L, 2),
			lua_tointeger(L, 3)
	);
	ex->refcount++;
	
	Exchange** pp = (Exchange**)lua_newuserdata(L, sizeof(Exchange**));
	
	*pp = ex;
	luaL_getmetatable(L, "MERCER.exchange");
	lua_setmetatable(L, -2);
	return 1;
}

int l_ex_gc(lua_State* L)
{
	Exchange* ex = checkExchange(L, 1);
	if(!ex) return 0;
	
	ex->refcount--;
	if(ex->refcount == 0)
		delete ex;
	
	return 0;
}

int l_ex_apply(lua_State* L)
{
	Exchange* ex = checkExchange(L, 1);
	SpinSystem* ss = checkSpinSystem(L, 2);
	
	if(!ex->apply(ss))
		return luaL_error(L, ex->errormsg.c_str());
	
	return 0;
}

int l_ex_addpath(lua_State* L)
{
	Exchange* ex = checkExchange(L, 1);
	if(!ex) return 0;

	if(lua_gettop(L) != 7 && lua_gettop(L) != 8)
		return luaL_error(L, "addPath(s1x, s1y, s1z, s2x, s2y, s2z, strength)");

	int s1x = lua_tointeger(L, 2)-1;
	int s1y = lua_tointeger(L, 3)-1;
	int s1z = lua_tointeger(L, 4)-1;

	int s2x = lua_tointeger(L, 5)-1;
	int s2y = lua_tointeger(L, 6)-1;
	int s2z = lua_tointeger(L, 7)-1;
	
	double strength = lua_isnumber(L, 8)?lua_tonumber(L, 8):1.0;
	int s1 = ex->getSite(s1x, s1y, s1z);
	int s2 = ex->getSite(s2x, s2y, s2z);

// 	printf("%i (%i %i %i)     %i (%i %i %i)    %g\n", s1, s1x, s1y, s1z, s2, s2x, s2y, s2z, strength);

	ex->addPath(s1, s2, strength);
	return 0;
}

int l_ex_member(lua_State* L)
{
	Exchange* ex = checkExchange(L, 1);
	if(!ex) return 0;

	int px = lua_tointeger(L, 2) - 1;
	int py = lua_tointeger(L, 3) - 1;
	int pz = lua_tointeger(L, 4) - 1;
	
	if(ex->member(px, py, pz))
		lua_pushboolean(L, 1);
	else
		lua_pushboolean(L, 0);

	return 1;
}

void registerExchange(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_ex_gc},
		{"apply",        l_ex_apply},
		{"addPath",      l_ex_addpath},
		{"member",       l_ex_member},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.exchange");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_ex_new},
		{NULL, NULL}
	};
		
	luaL_register(L, "Exchange", functions);
	lua_pop(L,1);	
}

