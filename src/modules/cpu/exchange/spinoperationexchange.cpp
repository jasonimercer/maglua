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
	: SpinOperation(nx, ny, nz, hash32(Exchange::typeName()))
{
	setSlotName("Exchange");
	size = 32;
	num  = 0;
	pathways = 0;
	pbc[0] = 1;
	pbc[1] = 1;
	pbc[2] = 1;

}

int Exchange::luaInit(lua_State* L)
{
	deinit();
	size = 32;
	num  = 0;
	pathways = (sss*)malloc(sizeof(sss) * size);
	
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	return 0;	
}

void Exchange::encode(buffer* b)
{
	SpinOperation::encode(b);
	char version = 0;
	encodeChar(version, b);
	
	encodeInteger(pbc[0], b);
	encodeInteger(pbc[1], b);
	encodeInteger(pbc[2], b);
	
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

	SpinOperation::decode(b);
	char version = decodeChar(b);
	if(version == 0)
	{
		pbc[0] = decodeInteger(b);
		pbc[1] = decodeInteger(b);
		pbc[2] = decodeInteger(b);
		
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
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
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

// This has a big change. As of April 21, 2013 the exchange 
// works with sigma, not M. This means that the site is now normalized.
// This change is required to ease cases when the site magnitude changes
// over time (LLB)
bool Exchange::apply(SpinSystem* ss)
{
	int slot = markSlotUsed(ss);

	dArray& hx = (*ss->hx[slot]);
	dArray& hy = (*ss->hy[slot]);
	dArray& hz = (*ss->hz[slot]);

	dArray& sx = (*ss->x);
	dArray& sy = (*ss->y);
	dArray& sz = (*ss->z);
	dArray& mm = (*ss->ms);

	#pragma omp parallel for shared(hx,sx)
	for(int i=0; i<num; i++)
	{
		const int t    = pathways[i].tosite;
		const int f    = pathways[i].fromsite;
		const double s = pathways[i].strength;
		
		if(mm[f])
			hx[t] += sx[f] * s * global_scale / mm[f];
	}
	#pragma omp parallel for shared(hy,sy)
	for(int i=0; i<num; i++)
	{
		const int t    = pathways[i].tosite;
		const int f    = pathways[i].fromsite;
		const double s = pathways[i].strength;
		
		if(mm[f])
			hy[t] += sy[f] * s * global_scale / mm[f];
	}
	#pragma omp parallel for shared(hz,sz)
	for(int i=0; i<num; i++)
	{
		const int t    = pathways[i].tosite;
		const int f    = pathways[i].fromsite;
		const double s = pathways[i].strength;
		
		if(mm[f])
			hz[t] += sz[f] * s * global_scale / mm[f];
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
			if(size > 1024*1024)
				size = size + 1024*1024;
			else
			{
				if(size == 0)
				{
					size = 32;
				}
				else
					size *= 8;
			}
// 			size *= 32;
// 			size++;
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

bool Exchange::getPath(int idx, int& fx, int& fy, int& fz, int& tx, int& ty, int& tz, double& strength)
{
	if(idx < 0 || idx >= numPaths())
		return false;
	
	idx2xyz(pathways[idx].fromsite, fx, fy, fz);
	idx2xyz(pathways[idx].tosite, tx, ty, tz);
	strength = pathways[idx].strength;
	return true;
}

int Exchange::mergePaths()
{
	sss* new_pathways = (sss*) malloc(sizeof(sss)*size);
	int new_num = 0;

	for(int i=0; i<num; i++)
	{
		int from = pathways[i].fromsite;
		int to   = pathways[i].tosite;
		
		if(from >= 0 && to >=0)
		{
			new_pathways[new_num].fromsite = from;
			new_pathways[new_num].tosite = to;
			new_pathways[new_num].strength = 0;
			
			for(int j=i; j<num; j++)
			{
				if(pathways[j].fromsite == from && pathways[j].tosite == to)
				{
					new_pathways[new_num].strength += pathways[j].strength;
					pathways[j].fromsite = -1; //remove from future searches
					pathways[j].tosite = -1; //remove from future searches
				}
			}
			new_num++;
		}
	}

	int delta = num - new_num;
	free(pathways);
	pathways = new_pathways;
	num = new_num;
	return delta;
}



static int l_addpath(lua_State* L)
{
	LUA_PREAMBLE(Exchange,ex,1);

	const int* pbc = ex->pbc;
	
	int r1, r2;
	int a[3];
	int b[3];
	
	r1 = lua_getNint(L, 3, a, 2,    1);
	if(r1<0)	return luaL_error(L, "invalid site");
	
	r2 = lua_getNint(L, 3, b, 2+r1, 1);
	if(r2<0)	return luaL_error(L, "invalid site");
	

	a[0]--; b[0]--;
	a[1]--; b[1]--;
	a[2]--; b[2]--;
	
	int nxyz[3];
	nxyz[0] = ex->nx;
	nxyz[1] = ex->ny;
	nxyz[2] = ex->nz;
	
	for(int i=0; i<3; i++)
	{
		if(pbc[i]) //then we will adjust to inside system if needed
		{
			while(a[i] < 0)
			{
				a[i] += 4*nxyz[i];
			}
			while(b[i] < 0)
			{
				b[i] += 4*nxyz[i];
			}
			a[i] %= nxyz[i];
			b[i] %= nxyz[i];
		}
		if(a[i] < 0 || a[i] >= nxyz[i])
			return 0;
		if(b[i] < 0 || b[i] >= nxyz[i])
			return 0;
	}
	
	int s1x = a[0];
	int s1y = a[1];
	int s1z = a[2];

	int s2x = b[0];
	int s2y = b[1];
	int s2z = b[2];

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

static int l_getPathsTo(lua_State* L)
{
	LUA_PREAMBLE(Exchange, ex, 1);
	
	int r1;
	int a[3];
	
	r1 = lua_getNint(L, 3, a, 2,    1);
	if(r1<0)	return luaL_error(L, "invalid site");
	
	int idx = ex->getidx(a[0]-1, a[1]-1, a[2]-1);

	lua_newtable(L);
	int j = 1;
	for(int i=0; i<ex->numPaths(); i++)
	{
		if(ex->pathways[i].tosite == idx)
		{
			lua_pushinteger(L, j);
			lua_pushinteger(L, i+1);
			lua_settable(L, -3);
			j++;
		}
	}
	
	return 1;
}
static int l_getPathsFrom(lua_State* L)
{
	LUA_PREAMBLE(Exchange, ex, 1);
	
	int r1;
	int a[3];
	
	r1 = lua_getNint(L, 3, a, 2,    1);
	if(r1<0)	return luaL_error(L, "invalid site");
	
	int idx = ex->getidx(a[0]-1, a[1]-1, a[2]-1);

	lua_newtable(L);
	int j = 1;
	for(int i=0; i<ex->numPaths(); i++)
	{
		if(ex->pathways[i].fromsite == idx)
		{
			lua_pushinteger(L, j);
			lua_pushinteger(L, i+1);
			lua_settable(L, -3);
			j++;
		}
	}
	
	return 1;
}


static int l_numberOfPaths(lua_State* L)
{
	LUA_PREAMBLE(Exchange, ex, 1);
	lua_pushinteger(L, ex->numPaths());
	return 1;
}

static int l_mergepaths(lua_State* L)
{
	LUA_PREAMBLE(Exchange, ex, 1);
	lua_pushinteger(L, ex->mergePaths());
	return 1;	
}

static int l_getPath(lua_State* L)
{
	LUA_PREAMBLE(Exchange, ex, 1);
	int idx = lua_tointeger(L, 2);
	
	int fx,fy,fz;
	int tx,ty,tz;
	double strength;
	
	if(!ex->getPath(idx-1, fx,fy,fz, tx,ty,tz, strength))
	{
		return luaL_error(L, "Invalid index");
	}
	
	lua_newtable(L);
	lua_pushinteger(L, 1);	lua_pushinteger(L, fx+1);	lua_settable(L, -3);
	lua_pushinteger(L, 2);	lua_pushinteger(L, fy+1);	lua_settable(L, -3);
	lua_pushinteger(L, 3);	lua_pushinteger(L, fz+1);	lua_settable(L, -3);

	lua_newtable(L);
	lua_pushinteger(L, 1);	lua_pushinteger(L, tx+1);	lua_settable(L, -3);
	lua_pushinteger(L, 2);	lua_pushinteger(L, ty+1);	lua_settable(L, -3);
	lua_pushinteger(L, 3);	lua_pushinteger(L, tz+1);	lua_settable(L, -3);

	lua_pushnumber(L, strength);
	return 3;
}

static int l_getpbc(lua_State* L)
{
	LUA_PREAMBLE(Exchange, ex, 1);
	lua_pushboolean(L, ex->pbc[0]);
	lua_pushboolean(L, ex->pbc[1]);
	lua_pushboolean(L, ex->pbc[2]);
	return 3;
}

static int l_setpbc(lua_State* L)
{
	LUA_PREAMBLE(Exchange, ex, 1);
	ex->pbc[0] = lua_toboolean(L, 2);
	ex->pbc[1] = lua_toboolean(L, 3);
	ex->pbc[2] = lua_toboolean(L, 4);
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
		lua_pushstring(L, "2 *3Vector*s, 1 Optional Number: The vectors define the lattice sites that share a pathway (as source site, destination site), the number is the strength of the pathway or 1 as a default. For example, if ex is an Exchange Operator then ex:addPath({1,1,1}, {1,1,2}, -1) and ex:addPath({1,1,2}, {1,1,1}, -1) would make two spins neighbours of each other with anti-ferromagnetic exchange.");
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
	
	if(func == l_numberOfPaths)
	{
		lua_pushstring(L, "Determine how many pathways exist in the operator");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Number of pathways");
		return 3;		
	}
		
	if(func == l_getPath)
	{
		lua_pushstring(L, "Get information about a path");
		lua_pushstring(L, "1 Integer: Index of path [1:numberOfPaths()]");
		lua_pushstring(L, "2 Tables, 1 Number: triplets of integers describing from and to sites. 1 number describing strength");
		return 3;		
	}
	
	if(func == l_getPathsTo)
	{
		lua_pushstring(L, "Get all path indices that connect to the given site");
		lua_pushstring(L, "1 *3Vector*: Index of to-site");
		lua_pushstring(L, "1 Tables: indices of paths that connect to the given site");
		return 3;			
	}
	if(func == l_getPathsFrom)
	{
		lua_pushstring(L, "Get all path indices that connect from the given site");
		lua_pushstring(L, "1 *3Vector*: Index of from-site");
		lua_pushstring(L, "1 Tables: indices of paths that connect from the given site");
		return 3;			
	}
	if(func == l_mergepaths)
	{
		lua_pushstring(L, "Combine repeated to-from pairs into a single path with combined strength");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;			
	}
	
	if(func == l_getpbc)
	{
		lua_pushstring(L, "Get the flags for periodicity in the X, Y and Z directions. Default true, true, true.");
		lua_pushstring(L, "");
		lua_pushstring(L, "3 Booleans: Each value corresponds to a cardinal direction. If true then new paths will use periodic boundaries for out-of-range sites otherwise the path will be ignored.");
		return 3;			
	}
	if(func == l_setpbc)
	{
		lua_pushstring(L, "Set the flags for periodicity in the X, Y and Z directions. Default true, true, true.");
		lua_pushstring(L, "3 Booleans: Each value corresponds to a cardinal direction. If true then new paths will use periodic boundaries for out-of-range sites otherwise the path will be ignored.");
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
		{"numberOfPaths",l_numberOfPaths},
		{"path",         l_getPath},
		{"pathsTo",      l_getPathsTo},
		{"pathsFrom",    l_getPathsFrom},
		{"mergePaths",   l_mergepaths},
		{"periodicXYZ", l_getpbc},
		{"setPeriodicXYZ", l_setpbc},
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



