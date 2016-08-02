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
	: SpinOperation(nx, ny, nz, hash32(Anisotropy::typeName()))
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
	newDataFromScript = false;
	
	ops = 0;
	size = 0;
	
	init();
}

int Anisotropy::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	size = 0;
	init();
	return 0;
}

Anisotropy::~Anisotropy()
{
	deinit();
	unregisterWS();
}

void Anisotropy::init()
{
	num = 0;
	if(size < 0)
		size = 1;

	ops = (ani*)malloc(sizeof(ani) * size);
	
	
	make_host();
}

void Anisotropy::deinit()
{
	if(ops)
	{
		free(ops);
	}
	size = 0;
	ops = 0;
	
	delete_host();
	delete_compressed();
	delete_uncompressed();
}

bool Anisotropy::make_host()
{
	if(h_nx)
		return true;

	malloc_host(&(h_nx), sizeof(double) * nxyz);
	malloc_host(&(h_ny), sizeof(double) * nxyz);
	malloc_host(&(h_nz), sizeof(double) * nxyz);
	malloc_host(&(h_k),  sizeof(double) * nxyz);

	for(int i=0; i<nxyz; i++)
	{
		h_nx[i] = 0;
		h_ny[i] = 0;
		h_nz[i] = 0;
		h_k[i] = 0;
	}
	return true;
}


static bool myfunction(Anisotropy::ani* i,Anisotropy::ani* j)
{
	return (i->site<j->site);
}

#include <algorithm>    // std::sort
#include <vector>       // std::vector
using namespace std;
// this is messy but more efficient than before
int Anisotropy::merge()
{
	if(num == 0)
		return 0;
	
	int original_number = num;
	
	vector<ani*> new_ops;
	
	for(int i=0; i<num; i++)
	{
		new_ops.push_back(&ops[i]);
	}
	sort (new_ops.begin(), new_ops.end(), myfunction);

	ani* new_ops2 = (ani*) malloc(sizeof(ani)*size);
	int new_num2 = num;
	
	for(unsigned int i=0; i<new_ops.size(); i++)
	{
		memcpy(&new_ops2[i], new_ops[i], sizeof(ani));
	}
	
	int current = 0;
	
	num = 1;
	
	// put in the 1st site
	memcpy(&ops[0], &new_ops2[0], sizeof(ani));
	
	for(int i=1; i<new_num2; i++)
	{
		if(new_ops2[i].site == ops[current].site)
		{
			ops[current].strength += new_ops2[i].strength;
		}
		else
		{
			current++;
			num++;
			memcpy(&ops[current], &new_ops2[i], sizeof(ani));
		}
	}
	
	

	int delta = original_number - num;
	free(new_ops2);
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

static bool sani_sort(const sani& d1, const sani& d2)
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

static bool sani_same(const sani& d1, const sani& d2)
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
		free_host(h_nx);
		free_host(h_ny);
		free_host(h_nz);
		free_host(h_k);
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

// convert from cpu style info to gpu precursor style info
void Anisotropy::writeToMemory()
{
	if(!newDataFromScript)
		return;
	newDataFromScript = false;

	make_uncompressed();
	merge();

	for(int i=0; i<num; i++)
	{
		const int site = ops[i].site;
		double nx = ops[i].axis[0];
		double ny = ops[i].axis[1];
		double nz = ops[i].axis[2];
		double K = ops[i].strength;

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
	newDataFromScript = true;
}


void Anisotropy::encode(buffer* b)
{
	SpinOperation::encode(b); //nx,ny,nz,global_scale
	char version = 0;
	encodeChar(version, b);
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
	char version = decodeChar(b);
	
	if(version == 0)
	{
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
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}
	return 0;
}

#if 0
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
#endif

bool Anisotropy::apply(SpinSystem* ss)
{
	SpinSystem* sss[1];
	sss[0] = ss;
	return apply(sss, 1);
}


bool Anisotropy::apply(SpinSystem** sss, int n)
{
	vector<int> slots;
	for(int i=0; i<n; i++)
		slots.push_back(markSlotUsed(sss[i]));

	writeToMemory();
	if(!make_compressed())
		make_uncompressed();
	
	const double** d_sx_N = (const double**)getVectorOfVectors(sss, n, "SpinOperation::apply_1", 's', 'x');
	const double** d_sy_N = (const double**)getVectorOfVectors(sss, n, "SpinOperation::apply_2", 's', 'y');
	const double** d_sz_N = (const double**)getVectorOfVectors(sss, n, "SpinOperation::apply_3", 's', 'z');

	      double** d_hx_N = getVectorOfVectors(sss, n, "SpinOperation::apply_4", 'h', 'x', &(slots[0]));
	      double** d_hy_N = getVectorOfVectors(sss, n, "SpinOperation::apply_5", 'h', 'y', &(slots[0]));
	      double** d_hz_N = getVectorOfVectors(sss, n, "SpinOperation::apply_6", 'h', 'z', &(slots[0]));
	
	if(compressed)
	{
		// d_LUT is non-null (since compressed)
		cuda_anisotropy_compressed_N(
			global_scale,
			d_sx_N, d_sy_N, d_sz_N,
			d_LUT, d_idx, 
			d_hx_N, d_hy_N, d_hz_N,
			nxyz, n);
	}
	else
	{
		if(!d_nx)
			make_uncompressed();
		
		cuda_anisotropy_N(
			global_scale,
			d_sx_N, d_sy_N, d_sz_N,
			d_nx, d_ny, d_nz, d_k,
			d_hx_N, d_hy_N, d_hz_N,
			nxyz, n);
	}
	
	for(int i=0; i<n; i++)
	{
		int slot = slots[i];
		sss[i]->hx[slot]->new_device = true;
		sss[i]->hy[slot]->new_device = true;
		sss[i]->hz[slot]->new_device = true;
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
ANISOTROPYCUDA_API int lib_register(lua_State* L);
ANISOTROPYCUDA_API int lib_version(lua_State* L);
ANISOTROPYCUDA_API const char* lib_name(lua_State* L);
ANISOTROPYCUDA_API int lib_main(lua_State* L);
}

#include "spinoperationanisotropy_uniaxial.h"
#include "spinoperationanisotropy_cubic.h"
ANISOTROPYCUDA_API int lib_register(lua_State* L)
{
	luaT_register<Anisotropy>(L);
	luaT_register<AnisotropyUniaxial>(L);
	luaT_register<AnisotropyCubic>(L);
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

