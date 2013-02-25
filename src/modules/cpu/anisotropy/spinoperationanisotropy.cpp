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

#include "spinoperationanisotropy.h"
#include "spinsystem.h"

#include <stdlib.h>

Anisotropy::Anisotropy(int nx, int ny, int nz)
	: SpinOperation(Anisotropy::typeName(), ANISOTROPY_SLOT, nx, ny, nz, hash32(Anisotropy::typeName()))
{
	ops = 0;
	//size = nx*ny*nz;
	size = 0;
	init();
}

int Anisotropy::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	size = 0;
// 	size = nx*ny*nz;
	init();
	return 0;
}

void Anisotropy::init()
{
	num = 0;
	if(size < 0)
		size = 1;

	ops = (ani*)malloc(sizeof(ani) * size);
}

void Anisotropy::deinit()
{
	if(ops)
	{
		free(ops);
	}
	size = 0;
	ops = 0;
}

int Anisotropy::merge()
{
	ani* new_ops = (ani*) malloc(sizeof(ani)*size);
	int new_num = 0;

	for(int i=0; i<num; i++)
	{
		int site = ops[i].site;
		
		if(site >= 0)
		{
			new_ops[new_num].site = site;
			new_ops[new_num].axis[0] = ops[i].axis[0];
			new_ops[new_num].axis[1] = ops[i].axis[1];
			new_ops[new_num].axis[2] = ops[i].axis[2];
			new_ops[new_num].strength = 0;
			
			for(int j=i; j<num; j++)
			{
				if(ops[j].site == site)
				{
					new_ops[new_num].strength += ops[j].strength;
					ops[j].site = -1; //remove from future searches
				}
			}
			new_num++;
		}
	}

	int delta = num - new_num;
	free(ops);
	ops = new_ops;
	num = new_num;
	return delta;
}


bool Anisotropy::getAnisotropy(int site, double& nx, double& ny, double& nz, double& K)
{
	for(int i=0; i<num; i++)
	{
		if(ops[i].site == site)
		{
			nx = ops[i].axis[0];
			ny = ops[i].axis[1];
			nz = ops[i].axis[2];
			K = ops[i].strength;
			return true;
		}
	}
	return false;
}


void Anisotropy::addAnisotropy(int site, double nx, double ny, double nz, double K)
{
	if(num == size)
	{
		if(size == 0)
			size = 32;
		else
			size = size * 2;
		ops = (ani*)realloc(ops, sizeof(ani) * size);
	}
	ops[num].site = site;
	ops[num].axis[0] = nx;
	ops[num].axis[1] = ny;
	ops[num].axis[2] = nz;
	ops[num].strength = K;
	num++;
}

void Anisotropy::encode(buffer* b)
{
	SpinOperation::encode(b); //nx,ny,nz,global_scale

	encodeInteger(num, b);
	for(int i=0; i<num; i++)
	{
		encodeInteger(ops[i].site, b);
		encodeDouble(ops[i].axis[0], b);
		encodeDouble(ops[i].axis[1], b);
		encodeDouble(ops[i].axis[2], b);
		encodeDouble(ops[i].strength, b);
	}
}

int Anisotropy::decode(buffer* b)
{
	deinit();
	SpinOperation::decode(b); //nx,ny,nz,global_scale
	
	num = decodeInteger(b);
	size = num;
	init();
	
	for(int i=0; i<size; i++)
	{
		const int site = decodeInteger(b);
		const double nx = decodeDouble(b);
		const double ny = decodeDouble(b);
		const double nz = decodeDouble(b);
		const double  K = decodeDouble(b);
		
		addAnisotropy(site, nx, ny, nz, K);
	}
	return 0;
}


Anisotropy::~Anisotropy()
{
	deinit();
}

bool Anisotropy::apply(SpinSystem* ss)
{
	markSlotUsed(ss);

	dArray& hx = (*ss->hx[slot]);
	dArray& hy = (*ss->hy[slot]);
	dArray& hz = (*ss->hz[slot]);

	dArray& x = (*ss->x);
	dArray& y = (*ss->y);
	dArray& z = (*ss->z);

	hx.zero();
	hy.zero();
	hz.zero();

	for(int j=0; j<num; j++)
	{
		const ani& op = ops[j];
		const int i = op.site;
		const double ms = (*ss->ms)[i];
		if(ms > 0)
		{
			const double SpinDotEasyAxis = 
								x[i] * op.axis[0] +
								y[i] * op.axis[1] +
								z[i] * op.axis[2];

			const double v = 2.0 * op.strength * SpinDotEasyAxis / (ms * ms);

			hx[i] += op.axis[0] * v * global_scale;
			hy[i] += op.axis[1] * v * global_scale;
			hz[i] += op.axis[2] * v * global_scale;
		}
	}
	return true;
}







static int l_get(lua_State* L)
{
	LUA_PREAMBLE(Anisotropy, ani, 1);

	double nx, ny, nz, K;

	int p[3];
	int r1 = lua_getNint(L, 3, p, 2, 1);

	if(r1<0)
		return luaL_error(L, "invalid site format");
	
	if(!ani->member(p[0]-1, p[1]-1, p[2]-1))
		return luaL_error(L, "site is not part of system");

	int idx = ani->getidx(p[0]-1, p[1]-1, p[2]-1);
	

	if(!ani->getAnisotropy(idx, nx, ny, nz, K))
	{
		lua_pushnumber(L, 1);
		lua_pushnumber(L, 0);
		lua_pushnumber(L, 0);
		lua_pushnumber(L, 0);
	}
	else
	{
		lua_pushnumber(L, nx);
		lua_pushnumber(L, ny);
		lua_pushnumber(L, nz);
		lua_pushnumber(L, K);
	}
	return 4;
}

static int l_numofax(lua_State* L)
{
	LUA_PREAMBLE(Anisotropy, ani, 1);
	lua_pushinteger(L, ani->num);
	return 1;
}


static int l_axisat(lua_State* L)
{
	LUA_PREAMBLE(Anisotropy, ani, 1);
	
	int idx = lua_tointeger(L, 2) - 1;

	if(idx < 0 || idx >= ani->num)
		return luaL_error(L, "Invalid axis index");
	

	const int site = ani->ops[idx].site;
	const double* axis = ani->ops[idx].axis;
	const double strength = ani->ops[idx].strength;
	
	int x,y,z;
	ani->idx2xyz(site, x, y, z);

	lua_newtable(L);
	lua_pushinteger(L, 1); lua_pushinteger(L, x+1); lua_settable(L, -3);
	lua_pushinteger(L, 2); lua_pushinteger(L, y+1); lua_settable(L, -3);
	lua_pushinteger(L, 3); lua_pushinteger(L, z+1); lua_settable(L, -3);
	
	lua_newtable(L);
	lua_pushinteger(L, 1); lua_pushnumber(L, axis[0]); lua_settable(L, -3);
	lua_pushinteger(L, 2); lua_pushnumber(L, axis[1]); lua_settable(L, -3);
	lua_pushinteger(L, 3); lua_pushnumber(L, axis[2]); lua_settable(L, -3);
	
	lua_pushnumber(L, strength);
	
	return 3;
}

static int l_add(lua_State* L)
{
	LUA_PREAMBLE(Anisotropy, ani, 1);

	int p[3];

	int r1 = lua_getNint(L, 3, p, 2, 1);
	
	if(r1<0)
		return luaL_error(L, "invalid site format");
	
	if(!ani->member(p[0]-1, p[1]-1, p[2]-1))
		return luaL_error(L, "site (%d, %d, %d) is not part of operator (%dx%dx%d)", p[0], p[1], p[2], ani->nx, ani->ny, ani->nz);

	int idx = ani->getidx(p[0]-1, p[1]-1, p[2]-1);

	double a[3];	
	int r2 = lua_getNdouble(L, 3, a, 2+r1, 0);
	if(r2<0)
		return luaL_error(L, "invalid anisotropy direction");

	/* anisotropy axis is a unit vector */
	const double lena = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
	
	if(lena > 0)
	{
		a[0] /= lena;
		a[1] /= lena;
		a[2] /= lena;
	}
	else
		return 0; //don't add ani
	
	double K = 0;

	if(lua_isnumber(L, 2+r1+r2))
		K = lua_tonumber(L, 2+r1+r2);
	else
		return luaL_error(L, "anisotropy needs strength");
	
	ani->addAnisotropy(idx, a[0], a[1], a[2], K);
	return 0;
}

static int l_mergeAxes(lua_State* L)
{
	LUA_PREAMBLE(Anisotropy, ani, 1);
	lua_pushinteger(L, ani->merge());
	return 1;	
}

int Anisotropy::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Computes the single ion anisotropy fields for a *SpinSystem*");
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	if(!lua_iscfunction(L, 1))
	{
		return luaL_error(L, "help expect zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
		
	if(func == l_add)
	{
		lua_pushstring(L, "Add a lattice site to the anisotropy calculation");
		lua_pushstring(L, "2 *3Vector*s, 1 number: The first *3Vector* defines a lattice site, the second defines an easy axis and is normalized. The number defines the strength of the Anisotropy.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_get)
	{
		lua_pushstring(L, "Fetch the anisotropy direction and magnitude at a given site.");
		lua_pushstring(L, "1 *3Vector*: The *3Vector* defines a lattice site.");
		lua_pushstring(L, "4 Numbers: The first 3 numbers define the normal axis, the 4th number is the magnitude.");
		return 3;
	}
	
	if(func == l_axisat)
	{
		lua_pushstring(L, "Return the site, easy axis and strength at the given index.");
		lua_pushstring(L, "1 Integer: Index of the axis.");
		lua_pushstring(L, "1 Table of 3 Integers, 1 Table of 3 Numbers, 1 Number: Coordinates of the site, direction of the easy axis and strength of the easy axis.");
		return 3;	
	}
	
	if(func == l_numofax)
	{
		lua_pushstring(L, "Return the number of easy axes in the operator");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Number of easy axes.");
		return 3;		
	}
	
	if(func == l_mergeAxes)
	{
		lua_pushstring(L, "Combine common site-axes into a single axis with a combined strength");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;			
	}
	
	
	return SpinOperation::help(L);
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Anisotropy::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, SpinOperation::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"add",          l_add},
		{"get",          l_get},
		{"numberOfAxes", l_numofax},
		{"axis", l_axisat},
		{"mergeAxes", l_mergeAxes},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}



#include "info.h"
extern "C"
{
ANISOTROPY_API int lib_register(lua_State* L);
ANISOTROPY_API int lib_version(lua_State* L);
ANISOTROPY_API const char* lib_name(lua_State* L);
ANISOTROPY_API int lib_main(lua_State* L);
}

ANISOTROPY_API int lib_register(lua_State* L)
{
	luaT_register<Anisotropy>(L);
	return 0;
}

ANISOTROPY_API int lib_version(lua_State* L)
{
	return __revi;
}

ANISOTROPY_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Anisotropy";
#else
	return "Anisotropy-Debug";
#endif
}

ANISOTROPY_API int lib_main(lua_State* L)
{
	return 0;
}
