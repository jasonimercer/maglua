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

#include "spinoperationexchange.h"
#include "spinsystem.h"

#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <algorithm>
#include <string.h>

Exchange::Exchange(int nx, int ny, int nz)
	: SpinOperation(Exchange::typeName(), EXCHANGE_SLOT, nx, ny, nz, hash32(Exchange::typeName()))
{
	size = 32;
	num  = 0;
	pathways = 0;
}

int Exchange::luaInit(lua_State* L)
{
	deinit();
	size = 32;
	num  = 0;
	pathways = (sss*)malloc(sizeof(sss) * size);
	
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
}

void Exchange::push(lua_State* L)
{
	luaT_push<Exchange>(L, this);
}


void Exchange::encode(buffer* b)
{
	encodeInteger(nx, b);
	encodeInteger(ny, b);
	encodeInteger(nz, b);
	
	encodeInteger(num, b);
	
	for(int i=0; i<num; i++)
	{
		encodeInteger(pathways[i].fromsite, b);
		encodeInteger(pathways[i].tosite, b);
		encodeDouble(pathways[i].strength, b);
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
	pathways = (sss*)malloc(sizeof(sss) * size);
	for(int i=0; i<num; i++)
	{
		pathways[i].fromsite = decodeInteger(b);
		pathways[i].tosite = decodeInteger(b);
		pathways[i].strength = decodeDouble(b);
	}
	return 0;
}

void Exchange::deinit()
{
	if(pathways)
	{
		free(pathways);
		pathways = 0;
	}
	num = 0;
}

Exchange::~Exchange()
{
	deinit();
}

bool Exchange::apply(SpinSystem* ss)
{
	markSlotUsed(ss);

	double* hx = ss->hx[slot];
	double* hy = ss->hy[slot];
	double* hz = ss->hz[slot];

	const double* sx = ss->x;
	const double* sy = ss->y;
	const double* sz = ss->z;

	#pragma omp parallel for shared(hx,sx)
	for(int i=0; i<num; i++)
	{
		const int t    = pathways[i].tosite;
		const int f    = pathways[i].fromsite;
		const double s = pathways[i].strength;
		
		hx[t] += sx[f] * s;
	}
	#pragma omp parallel for shared(hy,sy)
	for(int i=0; i<num; i++)
	{
		const int t    = pathways[i].tosite;
		const int f    = pathways[i].fromsite;
		const double s = pathways[i].strength;
		
		hy[t] += sy[f] * s;
	}
	#pragma omp parallel for shared(hz,sz)
	for(int i=0; i<num; i++)
	{
		const int t    = pathways[i].tosite;
		const int f    = pathways[i].fromsite;
		const double s = pathways[i].strength;
		
		hz[t] += sz[f] * s;
	}
	return true;
}

static bool mysort(Exchange::sss* i, Exchange::sss* j)
{
	if(i->tosite > j->tosite)
		return false;
	
	if(i->tosite == j->tosite)
		return i->fromsite < j->fromsite;
	
	return true;
}
	
// optimize the order of the sites
void Exchange::opt()
{
	return;
	// opt so that write and reads are ordered
	sss* p2 = (sss*)malloc(sizeof(sss) * size);
	memcpy(p2, pathways, sizeof(sss) * size);
	vector<sss*> vp;
	for(int i=0; i<num; i++)
	{
		vp.push_back(&p2[i]);
	}
	
	sort (vp.begin(), vp.end(), mysort);
	
	for(unsigned int i=0; i<vp.size(); i++)
	{
		pathways[i].tosite = vp[i]->tosite;
		pathways[i].fromsite = vp[i]->fromsite;
		pathways[i].strength = vp[i]->strength;
	}
	
	free(p2);
	return;
}

void Exchange::addPath(int site1, int site2, double str)
{
	if(str != 0)
	{
		if(num + 1 >= size)
		{
			size *= 2;
			size++;
			pathways = (sss*)realloc(pathways, sizeof(sss) * size);
			
			addPath(site1, site2, str);
			return;
		}
		
		pathways[num].fromsite = site1;
		pathways[num].tosite = site2;
		pathways[num].strength = str;
		num++;
	}
}




static int l_addpath(lua_State* L)
{
	LUA_PREAMBLE(Exchange,ex,1);

	int PBC = 1;
	if(lua_isboolean(L, -1))
	{
		PBC = lua_toboolean(L, -1);
	}
	
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
	
	if(!PBC)
	{
		if(!ex->member(s1x,s1y,s1z))
			return 0;
		if(!ex->member(s2x,s2y,s2z))
			return 0;
	}
	
	double strength = lua_isnumber(L, 2+r1+r2)?lua_tonumber(L, 2+r1+r2):1.0;
	int s1 = ex->getSite(s1x, s1y, s1z);
	int s2 = ex->getSite(s2x, s2y, s2z);

	ex->addPath(s1, s2, strength);
	return 0;
}


static int l_opt(lua_State* L)
{
	LUA_PREAMBLE(Exchange, ex, 1);
	ex->opt();
	return 0;
}


int Exchange::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates the exchange field of a *SpinSystem*");
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
		
	if(func == l_addpath)
	{
		lua_pushstring(L, "Add an exchange pathway between two sites.");
		lua_pushstring(L, "2 *3Vector*s, 1 Optional Number: The vectors define the lattice sites that share a pathway, the number is the strength of the pathway or 1 as a default. For example, if ex is an Exchange Operator then ex:addPath({1,1,1}, {1,1,2}, -1) and ex:addPath({1,1,2}, {1,1,1}, -1) would make two spins neighbours of each other with anti-ferromagnetic exchange.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_opt)
	{
		lua_pushstring(L, "Attempt to optimize the read/write patterns for exchange updates to minimize cache misses. Needs testing to see if it helps.");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}
	
	return SpinOperation::help(L);
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Exchange::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, SpinOperation::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"addPath",      l_addpath},
		{"add",          l_addpath},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}



#include "info.h"
extern "C"
{
EXCHANGE_API int lib_register(lua_State* L);
EXCHANGE_API int lib_version(lua_State* L);
EXCHANGE_API const char* lib_name(lua_State* L);
EXCHANGE_API int lib_main(lua_State* L);
}

EXCHANGE_API int lib_register(lua_State* L)
{
	luaT_register<Exchange>(L);
	return 0;
}

EXCHANGE_API int lib_version(lua_State* L)
{
	return __revi;
}

EXCHANGE_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Exchange";
#else
	return "Exchange-Debug";
#endif
}

EXCHANGE_API int lib_main(lua_State* L)
{
	return 0;
}



