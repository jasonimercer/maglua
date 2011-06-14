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
	: SpinOperation("Anisotropy", ANISOTROPY_SLOT, nx, ny, nz, ENCODE_ANISOTROPY)
{
	init();
}

void Anisotropy::init()
{
	nxyz = nx * ny * nz;
	ax = new double[nxyz];
	ay = new double[nxyz];
	az = new double[nxyz];
	strength = new double[nxyz];

	for(int i=0; i<nxyz; i++)
	{
		ax[i] = 0;
		ay[i] = 0;
		az[i] = 1;

		strength[i] = 1;
	}
}

void Anisotropy::deinit()
{
	if(ax)
	{
		delete [] ax;
		delete [] ay;
		delete [] az;
		delete [] strength;
	}
	ax = 0;
}

void Anisotropy::encode(buffer* b) const
{
	encodeInteger(nx, b);
	encodeInteger(ny, b);
	encodeInteger(nz, b);
	for(int i=0; i<nxyz; i++)
	{
		encodeDouble(ax[i], b);
		encodeDouble(ay[i], b);
		encodeDouble(az[i], b);
		encodeDouble(strength[i], b);
	}
}

int Anisotropy::decode(buffer* b)
{
	nx = decodeInteger(b);
	ny = decodeInteger(b);
	nz = decodeInteger(b);
	
	init();
	for(int i=0; i<nxyz; i++)
	{
		ax[i] = decodeDouble(b);
		ay[i] = decodeDouble(b);
		az[i] = decodeDouble(b);
		strength[i] = decodeDouble(b);
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

	double SpinDotEasyAxis;
	double v;
	
	double* hx = ss->hx[slot];
	double* hy = ss->hy[slot];
	double* hz = ss->hz[slot];
	
	#pragma omp parallel for shared(hx,hy,hz)
	for(int i=0; i<nxyz; i++)
	{
		const double ms = ss->ms[i];
		if(ms > 0)
		{
			const double
			SpinDotEasyAxis = ss->x[i] * ax[i] +
			                  ss->y[i] * ay[i] +
			                  ss->z[i] * az[i];

			const double v = strength[i] * SpinDotEasyAxis / (ms * ms);

// 			ss->hx[slot][i] = ax[i] * v;
// 			ss->hy[slot][i] = ay[i] * v;
// 			ss->hz[slot][i] = az[i] * v;
			hx[i] = ax[i] * v;
			hy[i] = ay[i] * v;
			hz[i] = az[i] * v;
		}
	}
	return true;
}







Anisotropy* checkAnisotropy(lua_State* L, int idx)
{
	Anisotropy** pp = (Anisotropy**)luaL_checkudata(L, idx, "MERCER.anisotropy");
    luaL_argcheck(L, pp != NULL, 1, "`Anisotropy' expected");
    return *pp;
}

void lua_pushAnisotropy(lua_State* L, Anisotropy* ani)
{
	ani->refcount++;
	
	Anisotropy** pp = (Anisotropy**)lua_newuserdata(L, sizeof(Anisotropy**));
	
	*pp = ani;
	luaL_getmetatable(L, "MERCER.anisotropy");
	lua_setmetatable(L, -2);
}

int l_ani_new(lua_State* L)
{
	int n[3];
	lua_getnewargs(L, n, 1);

	Anisotropy* ani = new Anisotropy(
			n[0], n[1], n[2]);
			
	lua_pushAnisotropy(L, ani);
	return 1;
}

int l_ani_gc(lua_State* L)
{
	Anisotropy* ani = checkAnisotropy(L, 1);
	if(!ani) return 0;
	
	ani->refcount--;
	if(ani->refcount == 0)
		delete ani;
	
	return 0;
}

int l_ani_apply(lua_State* L)
{
	Anisotropy* ani = checkAnisotropy(L, 1);
	SpinSystem* ss = checkSpinSystem(L, 2);
	
	if(!ani->apply(ss))
		return luaL_error(L, ani->errormsg.c_str());
	
	return 0;
}

int l_ani_member(lua_State* L)
{
	Anisotropy* ani = checkAnisotropy(L, 1);
	if(!ani) return 0;

	int px = lua_tointeger(L, 2) - 1;
	int py = lua_tointeger(L, 3) - 1;
	int pz = lua_tointeger(L, 4) - 1;
	
	if(ani->member(px, py, pz))
		lua_pushboolean(L, 1);
	else
		lua_pushboolean(L, 0);

	return 1;
}

int l_ani_set(lua_State* L)
{
	Anisotropy* ani = checkAnisotropy(L, 1);
	if(!ani) return 0;

	int p[3];
	
	int r1 = lua_getNint(L, 3, p, 2, 1);
	
	if(r1<0)
		return luaL_error(L, "invalid site format");
	
	if(!ani->member(p[0]-1, p[1]-1, p[2]-1))
		return luaL_error(L, "site is not part of system");

	int idx = ani->getidx(p[0]-1, p[1]-1, p[2]-1);

	double a[3];	
	int r2 = lua_getNdouble(L, 3, a, 2+r1, 0);
	if(r2<0)
		return luaL_error(L, "invalid anisotropy direction");
		
	ani->ax[idx] = a[0];
	ani->ay[idx] = a[1];
	ani->az[idx] = a[2];

	/* anisotropy axis is a unit vector */
	const double lena = 
		ani->ax[idx]*ani->ax[idx] +
		ani->ay[idx]*ani->ay[idx] +
		ani->az[idx]*ani->az[idx];
	
	if(lena > 0)
	{
		ani->ax[idx] /= sqrt(lena);
		ani->ay[idx] /= sqrt(lena);
		ani->az[idx] /= sqrt(lena);
	}
	else
		return 0; //don't add ani
		
	if(lua_isnumber(L, 2+r1+r2))
		ani->strength[idx] = lua_tonumber(L, 2+r1+r2);
	else
		return luaL_error(L, "anisotropy needs strength");

	return 0;
}


int l_ani_tostring(lua_State* L)
{
	Anisotropy* ani = checkAnisotropy(L, 1);
	if(!ani) return 0;
	
	lua_pushfstring(L, "Anisotropy (%dx%dx%d)", ani->nx, ani->ny, ani->nz);
	
	return 1;
}

static int l_ani_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.anisotropy");
	return 1;
}

static int l_ani_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Computes the single ion anisotropy fields for a *SpinSystem*");
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
	
	if(func == l_ani_new)
	{
		lua_pushstring(L, "Create a new Anisotropy Operator.");
		lua_pushstring(L, "1 *3Vector*: system size"); 
		lua_pushstring(L, "1 Anisotropy object");
		return 3;
	}
	

	if(func == l_ani_apply)
	{
		lua_pushstring(L, "Calculate the anisotropy of a *SpinSystem*");
		lua_pushstring(L, "1 *SpinSystem*: This system's Anisotropy field will be calculated based on the sites with Anisotropy.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_ani_set)
	{
		lua_pushstring(L, "Define a lattice site which has anisotropy");
		lua_pushstring(L, "2 *3Vector*s, 1 number: The first *3Vector* defines a lattice site, the second defines an easy axis and is normalized. The number defines the strength of the Anisotropy.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_ani_member)
	{
		lua_pushstring(L, "Determine if a lattice site is a member of the Operation.");
		lua_pushstring(L, "3 Integers: lattics site x, y, z.");
		lua_pushstring(L, "1 Boolean: True if x, y, z is part of the Operation, False otherwise.");
		return 3;
	}
	
	return 0;
}


void registerAnisotropy(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_ani_gc},
		{"__tostring",   l_ani_tostring},
		{"apply",        l_ani_apply},
		{"setSite",      l_ani_set},
		{"member",       l_ani_member},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.anisotropy");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_ani_new},
		{"help",                l_ani_help},
		{"metatable",           l_ani_mt},
		{NULL, NULL}
	};
		
	luaL_register(L, "Anisotropy", functions);
	lua_pop(L,1);	
}

