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
#include "spinoperationanisotropy.hpp"
#include "spinsystem.h"

#include <stdlib.h>
#include <math.h>

Anisotropy::Anisotropy(int nx, int ny, int nz)
	: SpinOperation("Anisotropy", ANISOTROPY_SLOT, nx, ny, nz, ENCODE_ANISOTROPY)
{
	d_nx = 0;
	init();
}

void Anisotropy::sync_dh(bool force)
{
	if(new_device || force)
	{
		ss_copyDeviceToHost(h_nx, d_nx, nxyz);
		ss_copyDeviceToHost(h_ny, d_ny, nxyz);
		ss_copyDeviceToHost(h_nz, d_nz, nxyz);
		ss_copyDeviceToHost(h_k,  d_k,  nxyz);
		
		new_host = false;
		new_device = false;
	}	
}

void Anisotropy::sync_hd(bool force)
{
	if(new_host || force)
	{
		ss_copyHostToDevice(d_nx, h_nx, nxyz);
		ss_copyHostToDevice(d_ny, h_ny, nxyz);
		ss_copyHostToDevice(d_nz, h_nz, nxyz);
		ss_copyHostToDevice(d_k,  h_k,  nxyz);
		
		new_host = false;
		new_device = false;
	}	
}

void Anisotropy::init()
{
	if(d_nx)
		deinit();
	nxyz = nx*ny*nz;
	
	ss_d_make3DArray(&d_nx, nx, ny, nz);
	ss_d_make3DArray(&d_ny, nx, ny, nz);
	ss_d_make3DArray(&d_nz, nx, ny, nz);
	ss_d_make3DArray(&d_k,  nx, ny, nz);
	
	h_nx = new double[nxyz];
	h_ny = new double[nxyz];
	h_nz = new double[nxyz];
	h_k  = new double[nxyz];
	
	new_host = false;
	new_device = true;
	
	ss_d_set3DArray(d_nx, nx, ny, nz, 0);
	ss_d_set3DArray(d_ny, nx, ny, nz, 0);
	ss_d_set3DArray(d_nz, nx, ny, nz, 0);
	ss_d_set3DArray(d_k,  nx, ny, nz, 0);
	
	sync_dh();
}

void Anisotropy::deinit()
{
	if(d_nx)
	{
		ss_d_free3DArray(d_nx);
		ss_d_free3DArray(d_ny);
		ss_d_free3DArray(d_nz);
		ss_d_free3DArray(d_k);

		delete [] h_nx;
		delete [] h_ny;
		delete [] h_nz;
		delete [] h_k;
	}
	d_nx = 0;
}

void Anisotropy::addAnisotropy(int site, double nx, double ny, double nz, double K)
{
	if(site >= 0 && site < nxyz)
	{
		double d = sqrt(nx*nx + ny*ny + nz*nz);

		if(d > 0)
		{
			sync_dh();

			h_nx[site] = nx/d;
			h_ny[site] = ny/d;
			h_nz[site] = nz/d;
			h_k[site]  = K;
		}

		new_host = true;
	}
}

void Anisotropy::encode(buffer* b)
{
	sync_dh();
	encodeInteger(nx, b);
	encodeInteger(ny, b);
	encodeInteger(nz, b);
	encodeInteger(nxyz, b);
	for(int i=0; i<nxyz; i++)
	{
		encodeInteger(i, b);
		encodeDouble(h_nx[i], b);
		encodeDouble(h_ny[i], b);
		encodeDouble(h_nz[i], b);
		encodeDouble(h_k[i],  b);
	}
}

int Anisotropy::decode(buffer* b)
{
	// some of the following seems like garbage. It's to
	// ensure compatibility with CPU version
	deinit();
	nx = decodeInteger(b);
	ny = decodeInteger(b);
	nz = decodeInteger(b);
	int num = decodeInteger(b);
	nxyz = nx*ny*nz;
	init();
	
	new_device = false;
	
	for(int i=0; i<num; i++)
	{
		int j = decodeInteger(b);
		const double nx = decodeDouble(b);
		const double ny = decodeDouble(b);
		const double nz = decodeDouble(b);
		const double  K = decodeDouble(b);
				
		addAnisotropy(j, nx, ny, nz, K);
	}
	
	//addAnisotropy marks new host
	return 0;
}

Anisotropy::~Anisotropy()
{
	deinit();
}

bool Anisotropy::apply(SpinSystem* ss)
{
	sync_hd();
	markSlotUsed(ss);

	double* d_hx = ss->d_hx[slot];
	double* d_hy = ss->d_hy[slot];
	double* d_hz = ss->d_hz[slot];

	cuda_anisotropy(
		ss->d_x, ss->d_y, ss->d_z, 
		d_nx, d_ny, d_nz, d_k,
		d_hx, d_hy, d_hz,
		nx, ny, nz);

	ss->new_device_fields[slot] = true;

	return true;
}







Anisotropy* checkAnisotropy(lua_State* L, int idx)
{
	Anisotropy** pp = (Anisotropy**)luaL_checkudata(L, idx, "MERCER.anisotropy");
    luaL_argcheck(L, pp != NULL, 1, "`Anisotropy' expected");
    return *pp;
}

void lua_pushAnisotropy(lua_State* L, Encodable* _ani)
{
	Anisotropy* ani = dynamic_cast<Anisotropy*>(_ani);
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

int l_ani_add(lua_State* L)
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
	
	
	
// 	ani->ax[idx] = a[0];
// 	ani->ay[idx] = a[1];
// 	ani->az[idx] = a[2];

	/* anisotropy axis is a unit vector */
	const double lena = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
// 		ani->ax[idx]*ani->ax[idx] +
// 		ani->ay[idx]*ani->ay[idx] +
// 		ani->az[idx]*ani->az[idx];
	
	if(lena > 0)
	{
		a[0] /= lena;
		a[1] /= lena;
		a[2] /= lena;
// 		ani->ax[idx] /= sqrt(lena);
// 		ani->ay[idx] /= sqrt(lena);
// 		ani->az[idx] /= sqrt(lena);
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
		lua_pushstring(L, "1 *3Vector*: system nxyz"); 
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
	
	if(func == l_ani_add)
	{
		lua_pushstring(L, "Add a lattice site to the anisotropy calculation");
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

static Encodable* newThing()
{
	return new Anisotropy;
}

void registerAnisotropy(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_ani_gc},
		{"__tostring",   l_ani_tostring},
		{"apply",        l_ani_apply},
		//{"setSite",      l_ani_set},
		{"add",          l_ani_add},
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
	Factory_registerItem(ENCODE_ANISOTROPY, newThing, lua_pushAnisotropy, "Anisotropy");
}


#include "info.h"
extern "C"
{
ANISOTROPYCUDA_API int lib_register(lua_State* L);
ANISOTROPYCUDA_API int lib_version(lua_State* L);
ANISOTROPYCUDA_API const char* lib_name(lua_State* L);
ANISOTROPYCUDA_API int lib_main(lua_State* L, int argc, char** argv);
}

ANISOTROPYCUDA_API int lib_register(lua_State* L)
{
	registerAnisotropy(L);
	return 0;
}

ANISOTROPYCUDA_API int lib_version(lua_State* L)
{
	return __revi;
}

ANISOTROPYCUDA_API const char* lib_name(lua_State* L)
{
	return "Anisotropy-Cuda";
}

ANISOTROPYCUDA_API int lib_main(lua_State* L, int argc, char** argv)
{
	return 0;
}
