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
#include <vector>
#include <algorithm>
using namespace std;

Anisotropy::Anisotropy(int nx, int ny, int nz)
	: SpinOperation("Anisotropy", ANISOTROPY_SLOT, nx, ny, nz, ENCODE_ANISOTROPY)
{
	d_nx = 0;
	d_ny = 0;
	d_nz = 0;
	d_k = 0;

	d_LUT = 0;
	d_idx = 0;
	h_nx = 0;
	new_host = true;
	registerWS();
	compressed = false;
	
	init();
}

Anisotropy::~Anisotropy()
{
	unregisterWS();
	deinit();
}

void Anisotropy::init()
{
	make_host();
}

void Anisotropy::deinit()
{
	delete_host();
	delete_compressed();
	delete_uncompressed();
}

bool Anisotropy::make_host()
{
	if(h_nx)
		return true;

	h_nx = new double[nxyz];
	h_ny = new double[nxyz];
	h_nz = new double[nxyz];
	h_k  = new double[nxyz];

	for(int i=0; i<nxyz; i++)
	{
		h_nx[i] = 0;
		h_ny[i] = 0;
		h_nz[i] = 0;
		h_k[i] = 0;
	}
	return true;
}


bool Anisotropy::make_uncompressed()
{
	delete_compressed();

	if(new_host || !d_nx)
	{
		delete_uncompressed();
		
		malloc_device(&d_nx, sizeof(double)*nxyz);
		malloc_device(&d_ny, sizeof(double)*nxyz);
		malloc_device(&d_nz, sizeof(double)*nxyz);
		malloc_device(&d_k,  sizeof(double)*nxyz);
		
		
		memcpy_h2d(d_nx, h_nx, sizeof(double)*nxyz);
		memcpy_h2d(d_ny, h_ny, sizeof(double)*nxyz);
		memcpy_h2d(d_nz, h_nz, sizeof(double)*nxyz);
		memcpy_h2d(d_k,  h_k,  sizeof(double)*nxyz);
	}
	new_host = false;
	compressed = false;
	return true;
}

class sani
{
public:
	sani(int s, double x, double y, double z, double k)
		: site(s), nx(x), ny(y), nz(z), K(k) {}
	sani(const sani& s) {site = s.site; nx=s.nx; ny=s.ny; nz=s.nz; K=s.K;}
	int site;
	double nx, ny, nz, K;
	char id;
};

bool sani_sort(const sani& d1, const sani& d2)
{
	if(d1.nx < d2.nx) return true;
	if(d1.nx > d2.nx) return false;
	
	if(d1.ny < d2.ny) return true;
	if(d1.ny > d2.ny) return false;
	
	if(d1.nz < d2.nz) return true;
	if(d1.nz > d2.nz) return false;
	
	if(d1.K  < d2.K ) return true;
	if(d1.K  > d2.K ) return false;
	
	return false;
}

bool sani_same(const sani& d1, const sani& d2)
{
	if(d1.nx != d2.nx)
		return false;
	if(d1.ny != d2.ny)
		return false;
	if(d1.nz != d2.nz)
		return false;
	if(d1.K  != d2.K )
		return false;
	return true;
}

bool Anisotropy::make_compressed()
{
	if(compressAttempted)
		return compressed;
		
	delete_compressed();
	delete_uncompressed();
	
	compressAttempted = true;
	if(!nxyz)
		return false;
	

	
	compressing = true;
	
	vector<sani> aa;
	for(int i=0; i<nxyz; i++)
		aa.push_back(sani(i, h_nx[i], h_ny[i], h_nz[i], h_k[i]));
	
	std::sort(aa.begin(), aa.end(), sani_sort);
	
	unsigned int last_one = 0;
	
	vector<unsigned int> uu; //uniques
	uu.push_back(0);
	aa[0].id = 0;
	
	for(unsigned int i=1; i<nxyz; i++)
	{
		if(!sani_same(aa[i], aa[last_one]))
		{
			last_one = i;
			uu.push_back(i);
			
		}
		aa[i].id = uu.size()-1;
	}
	
	if(uu.size() >= 255)
	{
		compressing = false;
		return false;
	}
	unique = uu.size();

	//ani can be compressed, build LUT
	double* h_LUT;// = new double[unique * 4];
	unsigned char* h_idx;
	
	malloc_host(&h_LUT, sizeof(double) * unique * 4);
	malloc_host(&h_idx, sizeof(unsigned char) * nxyz);
	
	for(unsigned int i=0; i<uu.size(); i++)
	{
		sani& q = aa[ uu[i] ];
		h_LUT[i*4+0] = q.nx;
		h_LUT[i*4+1] = q.ny;
		h_LUT[i*4+2] = q.nz;
		h_LUT[i*4+3] = q.K;
	}
	
	for(unsigned int i=0; i<nxyz; i++)
	{
		h_idx[i] = aa[i].id;
	}
	
	bool ok;
	ok  = malloc_device(&d_LUT, sizeof(double) * unique * 4);
	ok &= malloc_device(&d_idx, sizeof(unsigned char) * nxyz);
	
	if(!ok)
	{
		delete_compressed(); //incase LUT generated
		compressed = false;
		compressing = false;
		//should probably say something: this is bad.
		return false;
	}
	
	memcpy_h2d(d_LUT, h_LUT, sizeof(double) * unique * 4);
	memcpy_h2d(d_idx, h_idx, sizeof(unsigned char) * nxyz);
	
	free_host(h_LUT);
	free_host(h_idx);
	
	compressed = true;
	compressing = false;
	return true;
}


void Anisotropy::delete_host()
{
	if(h_nx)
	{
		delete [] h_nx;
		delete [] h_ny;
		delete [] h_nz;
		delete [] h_k;
		h_nx = 0;
		h_ny = 0;
		h_nz = 0;
		h_k  = 0;
	}	
}

void Anisotropy::delete_compressed()
{
	void** a[2] = {(void**)&d_LUT, (void**)&d_idx};
	for(int i=0; i<2; i++)
	{
		if(*a[i])
			free_device(*a[i]);
		*a[i] = 0;
	}
	compressed = false;
}

void Anisotropy::delete_uncompressed()
{
	void** a[4] = {
		(void**)&d_nx, (void**)&d_ny,
		(void**)&d_nz, (void**)&d_k
	};
	
	for(int i=0; i<4; i++)
	{
		if(*a[i])
			free_device(*a[i]);
		*a[i] = 0;
	}
}

void Anisotropy::addAnisotropy(int site, double nx, double ny, double nz, double K)
{
	if(site >= 0 && site < nxyz)
	{
		make_host();
		if(nz < 0)
		{
			nx *= -1.0;
			ny *= -1.0;
			nz *= -1.0;
		}
		
		double d = sqrt(nx*nx + ny*ny + nz*nz);

		if(d > 0)
		{
			h_nx[site] = nx/d;
			h_ny[site] = ny/d;
			h_nz[site] = nz/d;
			h_k[site]  = K;

			new_host = true;
			compressAttempted = false;

			delete_compressed();
			delete_uncompressed();
		}

	}
}

void Anisotropy::encode(buffer* b)
{
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



bool Anisotropy::apply(SpinSystem* ss)
{
	markSlotUsed(ss);
	ss->sync_spins_hd();

	double* d_hx = ss->d_hx[slot];
	double* d_hy = ss->d_hy[slot];
	double* d_hz = ss->d_hz[slot];

	if(!make_compressed())
		make_uncompressed();

	if(compressed)
	{
		// d_LUT is non-null (since compressed)
		cuda_anisotropy_compressed(
			ss->d_x, ss->d_y, ss->d_z,
			d_LUT, d_idx, 
			d_hx, d_hy, d_hz,
			nxyz);
	}
	else
	{
		if(!d_nx)
			make_uncompressed();
		
		cuda_anisotropy(
			ss->d_x, ss->d_y, ss->d_z, 
			d_nx, d_ny, d_nz, d_k,
			d_hx, d_hy, d_hz,
			nx, ny, nz);
	}
	

	ss->new_device_fields[slot] = true;

	return true;
}
 

bool Anisotropy::applyToSum(SpinSystem* ss)
{
//     double* d_wsAll = (double*)getWSMem(*3);

// 	double* d_wsx = d_wsAll + nxyz * 0;
//     double* d_wsy = d_wsAll + nxyz * 1;
//     double* d_wsz = d_wsAll + nxyz * 2;

	double* d_wsx;
	double* d_wsy;
	double* d_wsz;
	const int sz = sizeof(double)*nxyz;
	getWSMem(&d_wsx, sz, &d_wsy, sz, &d_wsz, sz);
	
//	markSlotUsed(ss);
	ss->sync_spins_hd();
	ss->ensureSlotExists(SUM_SLOT);

	if(!make_compressed())
		make_uncompressed();


	if(compressed)
	{
		// d_LUT is non-null (since compressed)
		cuda_anisotropy_compressed(
			ss->d_x, ss->d_y, ss->d_z,
			d_LUT, d_idx, 
			d_wsx, d_wsy, d_wsz,
			nxyz);
	}
	else
	{
		if(!d_nx)
			make_uncompressed();
		
		cuda_anisotropy(
			ss->d_x, ss->d_y, ss->d_z, 
			d_nx, d_ny, d_nz, d_k,
			d_wsx, d_wsy, d_wsz,
			nx, ny, nz);
	}
	

	const int nxyz = nx*ny*nz;
	cuda_addArrays(ss->d_hx[SUM_SLOT], nxyz, ss->d_hx[SUM_SLOT], d_wsx);
	cuda_addArrays(ss->d_hy[SUM_SLOT], nxyz, ss->d_hy[SUM_SLOT], d_wsy);
	cuda_addArrays(ss->d_hz[SUM_SLOT], nxyz, ss->d_hz[SUM_SLOT], d_wsz);
	ss->slot_used[SUM_SLOT] = true;

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

int l_ani_applytosum(lua_State* L)
{
	Anisotropy* ani = checkAnisotropy(L, 1);
	SpinSystem* ss = checkSpinSystem(L, 2);
	
	if(!ani->applyToSum(ss))
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

	if(func == l_ani_applytosum)
	{
		lua_pushstring(L, "Calculate the anisotropy of a *SpinSystem*");
		lua_pushstring(L, "1 *SpinSystem*: This system's Anisotropy field will be calculated based on the sites with Anisotropy and added to the total field slot.");
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
		{"applyToSum",   l_ani_applytosum},
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
ANISOTROPYCUDA_API int lib_main(lua_State* L);
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
#if defined NDEBUG || defined __OPTIMIZE__
	return "Anisotropy-Cuda";
#else
	return "Anisotropy-Cuda-Debug";
#endif
}

ANISOTROPYCUDA_API int lib_main(lua_State* L)
{
	return 0;
}
