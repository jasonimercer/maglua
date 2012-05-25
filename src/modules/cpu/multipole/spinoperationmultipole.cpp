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

#include "spinoperationmultipole.h"
#include "spinsystem.h"

#include "info.h"
#ifndef WIN32
#include <strings.h>
#endif

#include <stdlib.h>
#include <math.h>

Multipole::Multipole(int nx, int ny, int nz)
	: SpinOperation(Multipole::typeName(), DIPOLE_SLOT, nx, ny, nz, hash32(Multipole::typeName()))
{
	x = 0; y = 0; z = 0; weight = 0;
	oct = 0;
	init();
}

int Multipole::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	init();
	return 0;	
}

void Multipole::init()
{
	if(x) return;
	
	x = luaT_inc<dArray>(new dArray(nxyz,1,1));
	y = luaT_inc<dArray>(new dArray(nxyz,1,1));
	z = luaT_inc<dArray>(new dArray(nxyz,1,1));
	weight = luaT_inc<dArray>(new dArray(nxyz,1,1));
	
	int c = 0;
	for(int i=0; i<nx; i++)
	{
		for(int j=0; j<ny; j++)
		{
			for(int k=0; k<nz; k++)
			{
				(*x)[c] = i+1;
				(*y)[c] = j+1;
				(*z)[c] = k+1;
				(*weight)[c] = 1.0;
				c++;
			}
		}
	}
}
void Multipole::deinit()
{
	luaT_dec<dArray>(x);
	luaT_dec<dArray>(y);
	luaT_dec<dArray>(z);
	luaT_dec<dArray>(weight);
	x = 0; y = 0; z = 0; weight = 0;
	
	if(oct)
		delete oct;
	oct = 0;
}

	
void Multipole::push(lua_State* L)
{
	luaT_push<Multipole>(L, this);
}

void Multipole::encode(buffer* b)
{
// 	LongRange::encode(b);
}

int  Multipole::decode(buffer* b)
{
	deinit();
// 	LongRange::decode(b);
	
	return 0;
}

Multipole::~Multipole()
{
	deinit();
}

void Multipole::precompute()
{
	if(oct) return;
	
	oct = new OctTree(x, y, z);
 	oct->split(1);
	
}


bool Multipole::apply(SpinSystem* ss)
{

	return true;
}


static int l_mappos(lua_State* L)
{
	LUA_PREAMBLE(Multipole, mp, 1);
	
	int r1, r2;
	int site[3];
	double pos[3];
	
	r1 = lua_getNint(L, 3, site, 2, 1);
	if(r1 < 0)
		return luaL_error(L, "invalid site");
	
	r2 = lua_getNdouble(L, 3, pos, 2+r1, 0);
	if(r2 < 0)
		return luaL_error(L, "invalid position");
	
	double weight = 1.0;
	
	if(lua_isnumber(L, 2+r1+r2))
		weight = lua_tonumber(L, 2+r1+r2);
	
	int idx = mp->getidx(site[0]-1, site[1]-1, site[2]-1);
	
	(*(mp->x))[idx] = pos[0];
	(*(mp->y))[idx] = pos[1];
	(*(mp->z))[idx] = pos[2];
	(*(mp->weight))[idx] = weight;
	
	return 0;
}

static int l_getpos(lua_State* L)
{
	LUA_PREAMBLE(Multipole, mp, 1);
	
	int r1;
	int site[3];
	
	r1 = lua_getNint(L, 3, site, 2, 1);
	if(r1 < 0)
		return luaL_error(L, "invalid site");
	
	int idx = mp->getidx(site[0]-1, site[1]-1, site[2]-1);

	lua_pushnumber(L, (*(mp->x))[idx]);
	lua_pushnumber(L, (*(mp->y))[idx]);
	lua_pushnumber(L, (*(mp->z))[idx]);
	return 3;
}

int Multipole::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates the dipolar field of a *SpinSystem*");
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	if(!lua_iscfunction(L, 1))
	{
		return luaL_error(L, "help expect zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
		
	if(func == l_mappos)
	{
		lua_pushstring(L, "Map lattice coordinates to arbitrary positions in space");
		lua_pushstring(L, "1 *3Vector* (Integers), 1 *3Vector* (Numbers), [1 Number]: Index of site to map from, position in space to map to, optional weight of data (default 1.0).");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_getpos)
	{
		lua_pushstring(L, "Lookup a lattice mapping to get the position in space.");
		lua_pushstring(L, "1 *3Vector* (Integers): Index of site to.");
		lua_pushstring(L, "1 *3Vector* (Numbers): Position in space");
		return 3;
	}
	
	
	return SpinOperation::help(L);
}



static int l_pc(lua_State* L)
{
	LUA_PREAMBLE(Multipole, mp, 1);
	mp->precompute();
	return 0;
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Multipole::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, SpinOperation::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"mapPosition", l_mappos},
		{"getPosition", l_getpos},
		{"preCompute", l_pc},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}



extern "C"
{
MULTIPOLE_API int lib_register(lua_State* L);
MULTIPOLE_API int lib_version(lua_State* L);
MULTIPOLE_API const char* lib_name(lua_State* L);
MULTIPOLE_API int lib_main(lua_State* L);
}

MULTIPOLE_API int lib_register(lua_State* L)
{
	luaT_register<Multipole>(L);
	return 0;
}

MULTIPOLE_API int lib_version(lua_State* L)
{
	return __revi;
}


MULTIPOLE_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Multipole";
#else
	return "Multipole-Debug";
#endif
}

MULTIPOLE_API int lib_main(lua_State* L)
{
	return 0;
}


