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
#include "spinoperationexchange.hpp"
#include "spinsystem.h"

#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <algorithm>
#include <string.h>

Exchange::Exchange(int nx, int ny, int nz)
	: SpinOperation("Exchange", EXCHANGE_SLOT, nx, ny, nz, ENCODE_EXCHANGE)
{
	pathways = 0;
	
	d_strength = 0;
	d_fromsite = 0;
	maxFromSites = -1;
	h_strength = 0;
	h_fromsite = 0;
	
	new_host = true;
	
	init();
// 	fromsite =    (int*)malloc(sizeof(int)    * size);
// 	  tosite =    (int*)malloc(sizeof(int)    * size);
// 	strength = (double*)malloc(sizeof(double) * size);
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
	
	new_host = true;
	return 0;
}

void Exchange::sync()
{
	if(!new_host)
		return;
	new_host = false;
	opt();
	
	int oldMaxToSites = maxFromSites;
	
	//find out max number of neighbours
	int* nn = new int[nxyz];
	for(int i=0; i<nxyz; i++)
	{
		nn[i] = 0;
	}

	for(int i=0; i<num; i++)
	{
		nn[ pathways[i].tosite ]++;
	}
	
	maxFromSites = 0;
	
	for(int i=0; i<num; i++)
	{
		const int j = nn[ pathways[i].fromsite ];
		if(maxFromSites < j)
			maxFromSites = j;
	}
	
	// we will use nn to count number of recorded neighbours
	for(int i=0; i<nxyz; i++)
	{
		nn[i] = 0;
	}
	
	
	if(oldMaxToSites != maxFromSites)
	{
		if(d_strength)
		{
			ex_d_freeStrengthArray(d_strength);
			ex_d_freeNeighbourArray(d_fromsite);
			
			ex_h_freeStrengthArray(h_strength);
			ex_h_freeNeighbourArray(h_fromsite);
		}
		ex_d_makeStrengthArray(&d_strength, nx, ny, nz, maxFromSites);
		ex_d_makeNeighbourArray(&d_fromsite, nx, ny, nz, maxFromSites);
		
		ex_h_makeStrengthArray(&h_strength, nx, ny, nz, maxFromSites);
		ex_h_makeNeighbourArray(&h_fromsite, nx, ny, nz, maxFromSites);
	}
	
	for(int i=0; i<nxyz*maxFromSites; i++)
	{
		h_fromsite[i] = -1;
	}
	
	for(int i=0; i<num; i++)
	{
		int& j = pathways[i].fromsite;
		int& k = pathways[i].tosite;
		
		int& n = nn[k];
		
		h_fromsite[k*maxFromSites + n] = j;
		h_strength[k*maxFromSites + n] = pathways[i].strength;
		n++;
	}	
	delete [] nn;
	
	ex_hd_syncNeighbourArray(d_fromsite, h_fromsite, nx, ny, nz, maxFromSites);
	ex_hd_syncStrengthArray(d_strength, h_strength, nx, ny, nz, maxFromSites);
}
	
void Exchange::init()
{
	if(pathways)
		deinit();

	size = 32;
	num  = 0;
	pathways = (sss*)malloc(sizeof(sss) * size);

}
void Exchange::deinit()
{
	if(pathways)
	{
		free(pathways);
		
		if(d_strength)
		{
			ex_d_freeStrengthArray(d_strength);
			ex_d_freeNeighbourArray(d_fromsite);
			
			ex_h_freeStrengthArray(h_strength);
			ex_h_freeNeighbourArray(h_fromsite);
			
			d_strength = 0;
		}
		
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
	sync();
	
	double* d_hx = ss->d_hx[slot];
	double* d_hy = ss->d_hy[slot];
	double* d_hz = ss->d_hz[slot];

	const double* d_sx = ss->d_x;
	const double* d_sy = ss->d_y;
	const double* d_sz = ss->d_z;

	cuda_exchange(
		d_sx, d_sy, d_sz,
		d_strength, d_fromsite, maxFromSites,
		d_hx, d_hy, d_hz,
		nx, ny, nz);
	
	ss->new_device_fields[slot] = true;
	
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
// 	return;
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
		
		new_host = true;
	}
}



Exchange* checkExchange(lua_State* L, int idx)
{
	Exchange** pp = (Exchange**)luaL_checkudata(L, idx, "MERCER.exchange");
    luaL_argcheck(L, pp != NULL, 1, "`Exchange' expected");
    return *pp;
}

void lua_pushExchange(lua_State* L, Encodable* _ex)
{
	Exchange* ex = dynamic_cast<Exchange*>(_ex);
	if(!ex) return;
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

	bool PBC = true;
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


static int l_ex_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.exchange");
	return 1;
}

static int l_ex_tostring(lua_State* L)
{
	Exchange* ex = checkExchange(L, 1);
	if(!ex) return 0;
	
	lua_pushfstring(L, "Exchange (%dx%dx%d)", ex->nx, ex->ny, ex->nz);
	
	return 1;
}

static int l_ex_opt(lua_State* L)
{
	Exchange* ex = checkExchange(L, 1);
	if(!ex) return 0;
	
	ex->opt();
}

static int l_ex_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates the exchange field of a *SpinSystem*");
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
	
	if(func == l_ex_new)
	{
		lua_pushstring(L, "Create a new Exchange Operator.");
		lua_pushstring(L, "3 Integers: Defining the lattice dimensions"); 
		lua_pushstring(L, "1 Exchange object");
		return 3;
	}
	
	
	if(func == l_ex_apply)
	{
		lua_pushstring(L, "Calculate the exchange field of a *SpinSystem*");
		lua_pushstring(L, "1 *SpinSystem*: This spin system will receive the field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_ex_addpath)
	{
		lua_pushstring(L, "Add an exchange pathway between two sites.");
		lua_pushstring(L, "2 *3Vector*s, 1 Optional Number: The vectors define the lattice sites that share a pathway, the number is the strength of the pathway or 1 as a default. For example, if ex is an Exchange Operator then ex:addPath({1,1,1}, {1,1,2}, -1) and ex:addPath({1,1,2}, {1,1,1}, -1) would make two spins neighbours of each other with anti-ferromagnetic exchange.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_ex_member)
	{
		lua_pushstring(L, "Determine if a lattice site is a member of the Operation.");
		lua_pushstring(L, "3 Integers: lattics site x, y, z.");
		lua_pushstring(L, "1 Boolean: True if x, y, z is part of the Operation, False otherwise.");
		return 3;
	}
		
	if(func == l_ex_opt)
	{
		lua_pushstring(L, "Attempt to optimize the read/write patterns for exchange updates to minimize cache misses. Needs testing to see if it helps.");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}
	
	return 0;
}

static Encodable* newThing()
{
	return new Exchange;
}

void registerExchange(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_ex_gc},
		{"__tostring",   l_ex_tostring},
		{"apply",        l_ex_apply},
		{"addPath",      l_ex_addpath},
		{"add",          l_ex_addpath},
//		{"set",          l_ex_addpath},
		{"member",       l_ex_member},
//		{"optimize",     l_ex_opt},
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
		{"help",                l_ex_help},
		{"metatable",           l_ex_mt},
		{NULL, NULL}
	};
		
	luaL_register(L, "Exchange", functions);
	lua_pop(L,1);	
	Factory_registerItem(ENCODE_EXCHANGE, newThing, lua_pushExchange, "Exchange");
}

#include "info.h"
extern "C"
{
EXCHANGECUDA_API int lib_register(lua_State* L);
EXCHANGECUDA_API int lib_version(lua_State* L);
EXCHANGECUDA_API const char* lib_name(lua_State* L);
EXCHANGECUDA_API int lib_main(lua_State* L, int argc, char** argv);
}

EXCHANGECUDA_API int lib_register(lua_State* L)
{
	registerExchange(L);
	return 0;
}

EXCHANGECUDA_API int lib_version(lua_State* L)
{
	return __revi;
}

EXCHANGECUDA_API const char* lib_name(lua_State* L)
{
	return "Exchange-Cuda";
}

EXCHANGECUDA_API int lib_main(lua_State* L, int argc, char** argv)
{
	return 0;
}


