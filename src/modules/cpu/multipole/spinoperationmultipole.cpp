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
	: SpinOperation(nx, ny, nz, hash32(Multipole::typeName()))
{
	setSlotName("Multipole");
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
//	if(oct) return;
	
//	oct = new FMMOctTree(x, y, z);
// 	oct->split(1);
	
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


static int l_i2i(lua_State* L)
{
	//CORE_API int lua_getNdouble(lua_State* L, int N, double* vec, int pos, double def);
	double r[3];
	int pos = 2;
	int r1 = lua_getNdouble(L, 3, r, pos, 0);
	printf("r1 = %i\n", r1);
	monopole m(r);

	const int max_degree = 2;
	int cc = tensor_element_count(max_degree);

	printf("%f %f %f\n", m.r, m.t, m.p);
	printf("%f %f %f\n", m.x, m.y, m.z);
	printf("%f %f %f\n", r[0], r[1], r[2]);

	std::complex<double>* t = i2i_trans_mat(max_degree, m);

	const int n = cc;
	lua_newtable(L);
	for(int r=0; r<n; r++)
	{
		lua_pushinteger(L, r+1);
		lua_newtable(L);
		for(int c=0; c<n; c++)
		{
			lua_pushinteger(L, c+1);
			lua_newtable(L);
			lua_pushinteger(L, 1);
			lua_pushnumber(L, t[r*n+c].real());
			lua_settable(L, -3);
			lua_pushinteger(L, 2);
			lua_pushnumber(L, t[r*n+c].imag());
			lua_settable(L, -3);
			lua_settable(L, -3);
		}
		lua_settable(L, -3);
	}

	return 1;
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
		{"i2i", l_i2i},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}


#include "fmm_octtree.h"
#include "info.h"

extern "C"
{
MULTIPOLE_API int lib_register(lua_State* L);
MULTIPOLE_API int lib_version(lua_State* L);
MULTIPOLE_API const char* lib_name(lua_State* L);
MULTIPOLE_API int lib_main(lua_State* L);
}

#include "fmm_octtree_luafuncs.h"

static int l_getmetatable(lua_State* L)
{
    if(!lua_isstring(L, 1))
        return luaL_error(L, "First argument must be a metatable name");
    luaL_getmetatable(L, lua_tostring(L, 1));
    return 1;
}

MULTIPOLE_API int lib_register(lua_State* L)
{
	luaT_register<Multipole>(L);
	luaT_register<FMMOctTree>(L);


    lua_pushcfunction(L, l_getmetatable);
    lua_setglobal(L, "maglua_getmetatable");

    luaL_dofile_fmm_octtree_luafuncs(L);

    lua_pushnil(L);
    lua_setglobal(L, "maglua_getmetatable");


//	double x = -0.5;
//	for(int n=0; n<4; n++)
//	{
//		for(int l=-n; l<=n; l++)
//		{
//			printf("Pln(%i,%i,%g) = %g\n", n,l,x,Plm(n,l,x));
//		}
//	}

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


