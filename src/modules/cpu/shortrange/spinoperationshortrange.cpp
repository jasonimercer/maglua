/******************************************************************************
 * Copyright (C) 2008-2014 Jason Mercer.  All rights reserved.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 ******************************************************************************/

#include "spinoperationshortrange.h"
#include "spinsystem.h"

#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <algorithm>
#include <string.h>

ShortRange::ShortRange(int nx, int ny, int nz)
    : SpinOperation(nx, ny, nz, hash32(ShortRange::typeName()))
{
    setSlotName("ShortRange");
    size = 32;
    num  = 0;
    pathways = 0;
    pbc[0] = 1;
    pbc[1] = 1;
    pbc[2] = 1;

}

int ShortRange::luaInit(lua_State* L)
{
    deinit();
    size = 32;
    num  = 0;
    pathways = (sss*)malloc(sizeof(sss) * size);
	
    SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
    return 0;	
}

void ShortRange::encode(buffer* b)
{
    ENCODE_PREAMBLE
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
	for(int j=0; j<9; j++)
	    encodeDouble(pathways[i].matrix[j], b);
	encodeDouble(pathways[i].sig_dot_sig_pow, b);
		    
    }
}

int  ShortRange::decode(buffer* b)
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
	    for(int j=0; j<9; j++)
	    {
		pathways[i].matrix[j] = decodeDouble(b);
	    }
	    pathways[i].sig_dot_sig_pow = decodeDouble(b);
	}
    }
    else
    {
	fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
    }

    return 0;
}

void ShortRange::deinit()
{
    if(pathways)
    {
	free(pathways);
	pathways = 0;
    }
    num = 0;
}

ShortRange::~ShortRange()
{
    deinit();
}

bool ShortRange::apply(SpinSystem* ss)
{
    int slot = markSlotUsed(ss);

    /*
    dArray& hx = (*ss->hx[slot]);
    dArray& hy = (*ss->hy[slot]);
    dArray& hz = (*ss->hz[slot]);
    
    dArray& sx = (*ss->x);
    dArray& sy = (*ss->y);
    dArray& sz = (*ss->z);
    */

    double* hx = ss->hx[slot]->data();
    double* hy = ss->hy[slot]->data();
    double* hz = ss->hz[slot]->data();

    double* sx = ss->x->data();
    double* sy = ss->y->data();
    double* sz = ss->z->data();


    for(int i=0; i<num; i++)
    {
	const int& t    = pathways[i].tosite;
	const int& f    = pathways[i].fromsite;
	const double& s = pathways[i].strength;
	const double* m9 = pathways[i].matrix;
	const double& p = pathways[i].sig_dot_sig_pow;

	const double sds = sx[f]*sx[f] + sy[f]*sy[f] + sz[f]*sz[f];
	if(sds)
	{
	    const double sds_pow = pow(sds, p);
	    const double scale = s * global_scale * sds_pow;

	    hx[t] += (m9[0] * sx[f] + m9[1] * sy[f] + m9[2] * sz[f])* scale;
	    hy[t] += (m9[3] * sx[f] + m9[4] * sy[f] + m9[5] * sz[f])* scale;
	    hz[t] += (m9[6] * sx[f] + m9[7] * sy[f] + m9[8] * sz[f])* scale;
	}
    }
    return true;
}

void ShortRange::addPath(const int site1, const int site2, const double strength, const double* Lab_3x3, const double sig_dot_sig_pow)
{
    if(strength != 0)
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
			
	    addPath(site1, site2, strength, Lab_3x3, sig_dot_sig_pow);
	    return;
	}
	
	pathways[num].fromsite = site1;
	pathways[num].tosite = site2;
	pathways[num].strength = strength;
	memcpy(pathways[num].matrix, Lab_3x3, sizeof(double)*9);
	pathways[num].sig_dot_sig_pow = sig_dot_sig_pow;

	num++;
    }
}

bool ShortRange::getPath(int idx, int& fx, int& fy, int& fz, int& tx, int& ty, int& tz, double& strength, double* m9, double& sdsp)
{
    if(idx < 0 || idx >= numPaths())
	return false;

    idx2xyz(pathways[idx].fromsite, fx, fy, fz);
    idx2xyz(pathways[idx].tosite, tx, ty, tz);
    strength = pathways[idx].strength;
    sdsp = pathways[idx].sig_dot_sig_pow;
    memcpy(m9, pathways[idx].matrix, sizeof(double)*9);
    return true;
}






static int l_addpath(lua_State* L)
{
    LUA_PREAMBLE(ShortRange,sr,1);

    const int* pbc = sr->pbc;
	
    int r1, r2;
    int a[3];
    int b[3];
    double matrix[9];

    r1 = lua_getNint(L, 3, a, 2,    1);
    if(r1<0)	return luaL_error(L, "invalid site");
	
    r2 = lua_getNint(L, 3, b, 2+r1, 1);
    if(r2<0)	return luaL_error(L, "invalid site");
	

    a[0]--; b[0]--;
    a[1]--; b[1]--;
    a[2]--; b[2]--;
	
    int nxyz[3];
    nxyz[0] = sr->nx;
    nxyz[1] = sr->ny;
    nxyz[2] = sr->nz;
	
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

    double strength = lua_tonumber(L, 2+r1+r2);

    int s1 = sr->getSite(s1x, s1y, s1z);
    int s2 = sr->getSite(s2x, s2y, s2z);

    for(int i=0; i<9; i++)
    {
	lua_pushinteger(L, i+1);
	lua_gettable(L, 3+r1+r2);
	matrix[i] = lua_tonumber(L, -1);
	lua_pop(L, 1);
    }

    double sdsp = lua_tonumber(L, 4+r1+r2);

    sr->addPath(s1, s2, strength, matrix, sdsp);
    return 0;
}


static int l_getPathsTo(lua_State* L)
{
    LUA_PREAMBLE(ShortRange, sr, 1);
	
    int r1;
    int a[3];
	
    r1 = lua_getNint(L, 3, a, 2,    1);
    if(r1<0)	return luaL_error(L, "invalid site");
	
    int idx = sr->getidx(a[0]-1, a[1]-1, a[2]-1);

    lua_newtable(L);
    int j = 1;
    for(int i=0; i<sr->numPaths(); i++)
    {
	if(sr->pathways[i].tosite == idx)
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
    LUA_PREAMBLE(ShortRange, sr, 1);
	
    int r1;
    int a[3];
	
    r1 = lua_getNint(L, 3, a, 2,    1);
    if(r1<0)	return luaL_error(L, "invalid site");
	
    int idx = sr->getidx(a[0]-1, a[1]-1, a[2]-1);

    lua_newtable(L);
    int j = 1;
    for(int i=0; i<sr->numPaths(); i++)
    {
	if(sr->pathways[i].fromsite == idx)
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
    LUA_PREAMBLE(ShortRange, sr, 1);
    lua_pushinteger(L, sr->numPaths());
    return 1;
}

static int l_getPath(lua_State* L)
{
    LUA_PREAMBLE(ShortRange, sr, 1);
    int idx = lua_tointeger(L, 2);
	
    int fx,fy,fz;
    int tx,ty,tz;
    double strength;
    double sdsp;
    double matrix[9];

    if(!sr->getPath(idx-1, fx,fy,fz, tx,ty,tz, strength, matrix, sdsp))
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

    lua_newtable(L);
    for(int i=0; i<9; i++)
    {
	lua_pushinteger(L, i+1);
	lua_pushnumber(L, matrix[i]);
	lua_settable(L, -3);
    }

    lua_pushnumber(L, sdsp);

    return 5;
}

static int l_getpbc(lua_State* L)
{
    LUA_PREAMBLE(ShortRange, sr, 1);
    lua_pushboolean(L, sr->pbc[0]);
    lua_pushboolean(L, sr->pbc[1]);
    lua_pushboolean(L, sr->pbc[2]);
    return 3;
}

static int l_setpbc(lua_State* L)
{
    LUA_PREAMBLE(ShortRange, sr, 1);
    sr->pbc[0] = lua_toboolean(L, 2);
    sr->pbc[1] = lua_toboolean(L, 3);
    sr->pbc[2] = lua_toboolean(L, 4);
    return 0;
}

int ShortRange::help(lua_State* L)
{
    lua_CFunction func = lua_tocfunction(L, 1);

    if(func == l_numberOfPaths)
    {
	lua_pushstring(L, "Determine how many source to destination pathways exist in the operator");
	lua_pushstring(L, "");
	lua_pushstring(L, "1 Integer: Number of pathways");
	return 3;		
    }
		
    if(func == l_getPath)
    {
	lua_pushstring(L, "Get information about a path");
	lua_pushstring(L, "1 Integer: Index of path [1:numberOfPaths()]");
	lua_pushstring(L, "2 Tables, 1 Number, 1 Table, 1 Number: 2 Triplets of integers describing from and to sites, 1 number describing strength, 1 Table of 9 numbers describing the GammaAB style matrix, 1 Number representing the exponent of the sigma dot sigma scale factor.");
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
const luaL_Reg* ShortRange::luaMethods()
{
    if(m[127].name)return m;

    merge_luaL_Reg(m, SpinOperation::luaMethods());
    static const luaL_Reg _m[] =
	{
	    {"_addPath",      l_addpath},
	    // {"_add",          l_addpath},
	    {"numberOfPaths",l_numberOfPaths},
	    {"path",         l_getPath},
	    {"pathsTo",      l_getPathsTo},
	    {"pathsFrom",    l_getPathsFrom},
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
    SHORTRANGE_API int lib_register(lua_State* L);
    SHORTRANGE_API int lib_version(lua_State* L);
    SHORTRANGE_API const char* lib_name(lua_State* L);
    SHORTRANGE_API int lib_main(lua_State* L);
}


#include "shortrange_luafuncs.h"

static int l_getmetatable(lua_State* L)
{
    if(!lua_isstring(L, 1))
	return luaL_error(L, "First argument must be a metatable name");
    luaL_getmetatable(L, lua_tostring(L, 1));
    return 1;
}

SHORTRANGE_API int lib_register(lua_State* L)
{
    luaT_register<ShortRange>(L);

    lua_pushcfunction(L, l_getmetatable);
    lua_setglobal(L, "maglua_getmetatable");

    if(luaL_dostringn(L, __shortrange_luafuncs(), "shortrange_luafuncs.lua"))
    {
	fprintf(stderr, "%s\n", lua_tostring(L, -1));
	return luaL_error(L, lua_tostring(L, -1));
    }

    lua_pushnil(L);
    lua_setglobal(L, "maglua_getmetatable");
}

SHORTRANGE_API int lib_version(lua_State* L)
{
    return __revi;
}

SHORTRANGE_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
    return "ShortRange";
#else
    return "ShortRange-Debug";
#endif
}

SHORTRANGE_API int lib_main(lua_State* L)
{
    return 0;
}



