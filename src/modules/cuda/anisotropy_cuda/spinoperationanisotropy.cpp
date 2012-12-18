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
	: SpinOperation(Anisotropy::typeName(), ANISOTROPY_SLOT, nx, ny, nz, hash32(Anisotropy::typeName()))
{
	registerWS();
	d_nx = 0;
	d_ny = 0;
	d_nz = 0;
	d_k = 0;

	d_LUT = 0;
	d_idx = 0;
	h_nx = 0;
	new_host = true;
	compressed = false;
	
	init();
}

int Anisotropy::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	init();
	return 0;
}

void Anisotropy::push(lua_State* L)
{
	luaT_push<Anisotropy>(L, this);
}


Anisotropy::~Anisotropy()
{
	deinit();
	unregisterWS();
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
	
	for(int i=1; i<nxyz; i++)
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
	
	for(int i=0; i<nxyz; i++)
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

bool Anisotropy::getAnisotropy(int site, double& nx, double& ny, double& nz, double& K)
{
	if(site >= 0 && site < nxyz)
	{
		nx = 	h_nx[site];
		ny = 	h_ny[site];
		nz = 	h_nz[site];
		K  =	h_k[site];
		return true;
	}
    return false;
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
	SpinOperation::encode(b); //x y z global_scale
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
	SpinOperation::decode(b);
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

	double* d_hx = ss->hx[slot]->ddata();
	double* d_hy = ss->hy[slot]->ddata();
	double* d_hz = ss->hz[slot]->ddata();

	if(!make_compressed())
		make_uncompressed();

	if(compressed)
	{
		// d_LUT is non-null (since compressed)
		cuda_anisotropy_compressed(
			global_scale,
			ss->x->ddata(), ss->y->ddata(), ss->z->ddata(),
			d_LUT, d_idx, 
			d_hx, d_hy, d_hz,
			nxyz);
	}
	else
	{
		if(!d_nx)
			make_uncompressed();
		
		cuda_anisotropy(
			global_scale,
			ss->x->ddata(), ss->y->ddata(), ss->z->ddata(),
			d_nx, d_ny, d_nz, d_k,
			d_hx, d_hy, d_hz,
			nx, ny, nz);
	}
	
	ss->hx[slot]->new_device = true;
	ss->hy[slot]->new_device = true;
	ss->hz[slot]->new_device = true;
	
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
	getWSMemD(&d_wsx, sz, hash32("SpinOperation::apply_1"));
	getWSMemD(&d_wsy, sz, hash32("SpinOperation::apply_2"));
	getWSMemD(&d_wsz, sz, hash32("SpinOperation::apply_3"));
	
// 	const int sz = sizeof(double)*nxyz;
// 	getWSMem3(&d_wsx, sz, &d_wsy, sz, &d_wsz, sz);
	
//	markSlotUsed(ss);
//	ss->sync_spins_hd();
	ss->ensureSlotExists(SUM_SLOT);

	if(!make_compressed())
		make_uncompressed();


	if(compressed)
	{
		// d_LUT is non-null (since compressed)
		cuda_anisotropy_compressed(
			global_scale,
			ss->x->ddata(), ss->y->ddata(), ss->z->ddata(),
			d_LUT, d_idx, 
			d_wsx, d_wsy, d_wsz,
			nxyz);
	}
	else
	{
		if(!d_nx)
			make_uncompressed();
		
		cuda_anisotropy(
			global_scale,
			ss->x->ddata(), ss->y->ddata(), ss->z->ddata(),
			d_nx, d_ny, d_nz, d_k,
			d_wsx, d_wsy, d_wsz,
			nx, ny, nz);
	}
	

	const int nxyz = nx*ny*nz;
	
	arraySumAll(ss->hx[SUM_SLOT]->ddata(), ss->hx[SUM_SLOT]->ddata(), d_wsx, nxyz);
	arraySumAll(ss->hy[SUM_SLOT]->ddata(), ss->hy[SUM_SLOT]->ddata(), d_wsy, nxyz);
	arraySumAll(ss->hz[SUM_SLOT]->ddata(), ss->hz[SUM_SLOT]->ddata(), d_wsz, nxyz);
	
	ss->hx[SUM_SLOT]->new_device = true;
	ss->hy[SUM_SLOT]->new_device = true;
	ss->hz[SUM_SLOT]->new_device = true;
	
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


static int l_add(lua_State* L)
{
	LUA_PREAMBLE(Anisotropy, ani, 1);

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
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
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
	luaT_register<Anisotropy>(L);
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

