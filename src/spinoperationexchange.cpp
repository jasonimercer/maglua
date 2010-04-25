#include "spinoperationexchange.h"
#include "spinsystem.h"

#include <stdlib.h>
#include <stdio.h>

Exchange::Exchange(int nx, int ny, int nz)
	: SpinOperation("Exchange", EXCHANGE_SLOT, nx, ny, nz, ENCODE_EXCHANGE)
{
	size = 32;
	num  = 0;

	fromsite =    (int*)malloc(sizeof(int)    * size);
	  tosite =    (int*)malloc(sizeof(int)    * size);
	strength = (double*)malloc(sizeof(double) * size);
}

void Exchange::encode(buffer* b) const
{
	encodeInteger(nx, b);
	encodeInteger(ny, b);
	encodeInteger(nz, b);
	
	encodeInteger(num, b);
	
	for(int i=0; i<num; i++)
	{
		encodeInteger(fromsite[i], b);
		encodeInteger(  tosite[i], b);
		encodeDouble(strength[i], b);
	}
}

int  Exchange::decode(buffer* b)
{
	deinit();

	nx = decodeInteger(b);
	ny = decodeInteger(b);
	nz = decodeInteger(b);
	nxyz = nx * ny * nz;
	
	size = decodeInteger(b);
	num = size;
	size++; //so we can double if size == 0
	fromsite =    (int*)malloc(sizeof(int)    * size);
	  tosite =    (int*)malloc(sizeof(int)    * size);
	strength = (double*)malloc(sizeof(double) * size);
	
	for(int i=0; i<num; i++)
	{
		fromsite[i] = decodeInteger(b);
		  tosite[i] = decodeInteger(b);
		strength[i] = decodeDouble(b);
	}
	return 0;
}

void Exchange::deinit()
{
	if(fromsite)
	{
		free(fromsite);
		free(  tosite);
		free(strength);
		fromsite = 0;
	}
}

Exchange::~Exchange()
{
	deinit();
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
		
		hx[t] += sx[f] * s;
		hy[t] += sy[f] * s;
		hz[t] += sz[f] * s;
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

void lua_pushExchange(lua_State* L, Exchange* ex)
{
	ex->refcount++;
	
	Exchange** pp = (Exchange**)lua_newuserdata(L, sizeof(Exchange**));
	
	*pp = ex;
	luaL_getmetatable(L, "MERCER.exchange");
	lua_setmetatable(L, -2);
}

int l_ex_new(lua_State* L)
{
	int n[3];
	lua_getnewargs(L, n, 1);

	lua_pushExchange(L, new Exchange(n[0], n[1], n[2]));
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

	int r1, r2;
	int a[3];
	int b[3];
	
	r1 = lua_getNint(L, 3, a, 2,    1);
	if(r1<0)	return luaL_error(L, "invalid site");
	
	r2 = lua_getNint(L, 3, b, 2+r1, 1);
	if(r2<0)	return luaL_error(L, "invalid site");
	

	int s1x = a[0]-1;
	int s1y = a[1]-1;
	int s1z = a[2]-1;

	int s2x = b[0]-1;
	int s2y = b[1]-1;
	int s2z = b[2]-1;
	
	printf("%i\n", 2+r1+r2);
	double strength = lua_isnumber(L, 2+r1+r2)?lua_tonumber(L, 2+r1+r2):1.0;
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

