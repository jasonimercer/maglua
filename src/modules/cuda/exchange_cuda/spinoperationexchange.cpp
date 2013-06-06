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
#include "spinoperationexchange.hpp"
#include "spinsystem.h"
#include "array_core_cuda.h"

#include <stdlib.h>
#include <stdio.h>

#include <vector>
#include <algorithm>
#include <iostream>
#include <string.h>
using namespace std;

Exchange::Exchange(int nx, int ny, int nz)
	: SpinOperation(nx, ny, nz, hash32(Exchange::typeName()))
{
	registerWS();
	pathways = 0;
	
	d_strength = 0;
	d_fromsite = 0;
	maxFromSites = -1;

	pbc[0] = 1;
	pbc[1] = 1;
	pbc[2] = 1;
	
	d_LUT = 0;
	d_idx = 0;
	compress_max_neighbours = 0;
	
	new_host = true;
	compressed = false;
	compressAttempted = false;
	init();
}

int Exchange::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	init();
	return 0;	
}

Exchange::~Exchange()
{
	deinit();
	unregisterWS();
}

void Exchange::init()
{
	make_host();
}

void Exchange::deinit()
{
	delete_host();
	delete_compressed();
	delete_uncompressed();
}

bool Exchange::make_uncompressed()
{
	if(!new_host)
		return true;
	
// 	printf("make uncompressed\n");
	
	new_host = false;
	
	
	//find out max number of neighbours
	int* nn = new int[nxyz];
	for(int i=0; i<nxyz; i++)
		nn[i] = 0;

	for(int i=0; i<num; i++)
		nn[ pathways[i].tosite ]++;
	
	maxFromSites = 0;
	for(int i=0; i<num; i++)
	{
		const int j = nn[ pathways[i].fromsite ];
		if(maxFromSites < j)
			maxFromSites = j;
	}
	
// 	printf("maxFromSites = %i\n", maxFromSites);
	
	// we will use nn to count number of recorded neighbours
	for(int i=0; i<nxyz; i++)
		nn[i] = 0;
	
	int* h_fromsite;
	double* h_strength;
	malloc_host(&h_fromsite, sizeof(int)*maxFromSites*nxyz);
	malloc_host(&h_strength, sizeof(double)*maxFromSites*nxyz);
	
	for(int i=0; i<nxyz*maxFromSites; i++)
	{
		h_fromsite[i] = 0;
		h_strength[i] = 0;
	}
	
	for(int i=0; i<num; i++)
	{
		const int j = pathways[i].fromsite;
		const int k = pathways[i].tosite;
		
		int& n = nn[k];
		
		h_fromsite[k*maxFromSites + n] = j;
		h_strength[k*maxFromSites + n] = pathways[i].strength;
		n++;
	}	
	delete [] nn;
	
	delete_uncompressed();
	//todo: put memchecks around the following. see if compressed is req'd
	malloc_device(&d_fromsite, sizeof(int)*maxFromSites*nxyz);
	malloc_device(&d_strength, sizeof(double)*maxFromSites*nxyz);
	
	memcpy_h2d(d_fromsite, h_fromsite, sizeof(int)*maxFromSites*nxyz);
	memcpy_h2d(d_strength, h_strength, sizeof(double)*maxFromSites*nxyz);
	
	free_host(h_fromsite);
	free_host(h_strength);

	return true;
}


class ex_comp
{
public:
	ex_comp(int ID=0) {lut_id = 0; id=ID;};
	ex_comp(const ex_comp& r) {
		id=r.id; 
		lut_id=r.lut_id;
		for(unsigned int i=0; i<r.from_off_strength.size(); i++)
			from_off_strength.push_back(r.from_off_strength[i]);
	}
	void sort();
	
	int id;
	int lut_id;
	int add(int from, double strength, int nxyz);
	vector< pair<int,double> > from_off_strength;
};

void ex_comp::sort()
{
	std::sort(from_off_strength.begin(), from_off_strength.end());
}


int ex_comp::add(int from, double strength, int nxyz)
{
	int offset = (from - id + nxyz) % nxyz;

	from_off_strength.push_back(pair<int,double>(offset,strength));
	
	return offset;
}


bool ex_comp_sort(const ex_comp& d1, const ex_comp& d2)
{
	const unsigned int s1 = d1.from_off_strength.size();
	const unsigned int s2 = d2.from_off_strength.size();
	
	if(s1 < s2) return true;
	if(s1 > s2) return false;
	
	for(unsigned int i=0; i<s1; i++)
	{
		const pair<int,double>& p1 = d1.from_off_strength[i];
		const pair<int,double>& p2 = d2.from_off_strength[i];
		
		if(p1 < p2) return true;
		if(p2 < p1) return false;
	}
	return false;
}

bool ex_comp_same(const ex_comp& d1, const ex_comp& d2)
{
	if(d1.from_off_strength.size() != d2.from_off_strength.size())
		return false;
	
	for(unsigned int i=0; i<d1.from_off_strength.size(); i++)
	{
		if(d1.from_off_strength[i].first != d2.from_off_strength[i].first)
			return false;
		if(d1.from_off_strength[i].second != d2.from_off_strength[i].second)
			return false;
	}
	
	return true;
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

	new_host = true;

	return delta;
}



// #define DEBUG_EX_COMP 1
bool Exchange::make_compressed()
{
	if(compressAttempted)
		return compressed;
	
	delete_compressed();
	
	vector<ex_comp> vec;

	for(int i=0; i<nxyz; i++)
	{
		vec.push_back(ex_comp(i));
	}
	
#ifdef DEBUG_EX_COMP
	printf("Building vector\n");
#endif
	for(int i=0; i<num; i++)
	{
		const int to = pathways[i].tosite;
		const int from = pathways[i].fromsite;
		const double strength = pathways[i].strength;

		const int offset = vec[to].add(from, strength, nxyz);
#if 0	
#ifdef DEBUG_EX_COMP
		printf("%5i %5i %4.3f (%+3i)\n", from, to, strength, offset);
#endif
#endif
	}
		
	
	
#ifdef DEBUG_EX_COMP
	printf("Sorting vector\n");
#endif
	for(int i=0; i<nxyz; i++)
		vec[i].sort();
	
	std::sort(vec.begin(), vec.end(), ex_comp_sort);

#if 0
#ifdef DEBUG_EX_COMP
	cout << "------------------------" << endl;
	for(int i=0; i<nxyz; i++)
	{
		for(int j=0; j<vec[i].from_off_strength.size(); j++)
		{
			printf("%6i", vec[i].from_off_strength[j].first);
// 			cout <<   << " ";
		}
		cout << endl;
	}
	cout << "------------------------" << endl;
	
	printf("calc max neighbours\n");
#endif
#endif
	compress_max_neighbours = 0;
	for(int i=0; i<nxyz; i++)
	{
		if((int)vec[i].from_off_strength.size() > compress_max_neighbours)
			compress_max_neighbours = vec[i].from_off_strength.size();
	}


#ifdef DEBUG_EX_COMP
	printf("find uniques\n");
#endif

	vec[0].lut_id = 0;
	int last_unique = 0;
	vector<int> uniques;
	uniques.push_back(0);
	for(int i=1; i<nxyz; i++)
	{
		if(ex_comp_same(vec[last_unique], vec[i]))
		{
			vec[i].lut_id = vec[last_unique].lut_id;
		}
		else
		{
			uniques.push_back(i);
			const int next_lut_id = vec[last_unique].lut_id + 1;
			last_unique = i;
			vec[i].lut_id = next_lut_id;
		}
	}
	
#ifdef DEBUG_EX_COMP
	printf("%i uniques\n", (int)(uniques.size()));
#endif
	//LUT can do up to 255 flavours
	if(uniques.size() >= 255) 
	{
		compressAttempted = true;
		compressed = false;
		return false;
	}

	ex_compressed_struct* h_LUT;
	unsigned char* h_idx;
	
	malloc_host(&h_LUT, sizeof(ex_compressed_struct) * uniques.size()*compress_max_neighbours+1);
	malloc_host(&h_idx, sizeof(unsigned char) * nxyz);
	
	malloc_device(&d_LUT, sizeof(ex_compressed_struct) * uniques.size()*compress_max_neighbours+1);
	malloc_device(&d_idx, sizeof(unsigned char) * nxyz);
	
	// build LUT
#ifdef DEBUG_EX_COMP
	printf("build LUT\n");
#endif

	for(unsigned int i=0; i<uniques.size(); i++)
	{
		const int j = uniques[i];
		const int js = vec[j].from_off_strength.size();
		for(int k=0; k<js; k++)
		{
			h_LUT[i*compress_max_neighbours + k].offset   = vec[j].from_off_strength[k].first;
			h_LUT[i*compress_max_neighbours + k].strength = vec[j].from_off_strength[k].second;
		}
		int dummy_offset = 0;
		if(js)
			dummy_offset = vec[j].from_off_strength[js-1].first;

		// padding out structure with zero strength dummy interactions:
		//  gets rid of if statements on device
		for(int k=js; k<compress_max_neighbours; k++)
		{
			h_LUT[i*compress_max_neighbours + k].offset   = dummy_offset;
			h_LUT[i*compress_max_neighbours + k].strength = 0.0;
		}
	}
#ifdef DEBUG_EX_COMP
	printf("LUT:\n");
	for(unsigned int i=0; i<uniques.size(); i++)
	{
		printf("LUT(%3i): ", i);
		for(int k=0; k<compress_max_neighbours; k++)
		{
			printf("%+4i %4.3f  ", h_LUT[i*compress_max_neighbours + k].offset, h_LUT[i*compress_max_neighbours + k].strength);
		}
		printf("\n");
	}

#endif

#ifdef DEBUG_EX_COMP
	printf("build idx\n");
#endif
	for(int i=0; i<nxyz; i++)
	{
		const int j = vec[i].id;
		h_idx[j] = vec[i].lut_id;
#if 0
#ifdef DEBUG_EX_COMP
		printf("%8i   %4i\n", j, h_idx[j]);
#endif
#endif		
	}
	
#ifdef DEBUG_EX_COMP
	printf("memcpy\n");
#endif
	memcpy_h2d(d_LUT, h_LUT, sizeof(ex_compressed_struct) * uniques.size()*compress_max_neighbours);
	memcpy_h2d(d_idx, h_idx, sizeof(unsigned char) * nxyz);

	free_host(h_LUT);
	free_host(h_idx);
	
	compressed = true;
	compressAttempted = true;	
// 	printf("done\n");
	return true;
}

bool Exchange::make_host()
{
	if(pathways)
		delete_host();

	size = 32;
	num  = 0;
	pathways = (sss*)malloc(sizeof(sss) * size);
	new_host = true;
	
	return true;
}

void Exchange::delete_uncompressed()
{
	void** a[2] = {(void**)&d_strength, (void**)&d_fromsite};
	for(int i=0; i<2; i++)
	{
		if(*a[i])
			free_device(*a[i]);
		*a[i] = 0;
	}
}


void Exchange::delete_compressed()
{
	if(compressed)
	{
		void** a[2] = {(void**)&d_LUT, (void**)&d_idx};
		for(int i=0; i<2; i++)
		{
			if(*a[i])
				free_device(*a[i]);
			*a[i] = 0;
		}
	}
	
	compressed = false;
}

void Exchange::delete_host()
{
	if(pathways)
	{
		free(pathways);	
		pathways = 0;
	}
	num = 0;
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
		
		compressAttempted = false;
		new_host = true;
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}
	return 0;
}

bool Exchange::apply(SpinSystem** sss, int n)
{
	vector<int> slots;
	
	for(int i=0; i<n; i++)
	{
		int slot = markSlotUsed(sss[i]);
		slots.push_back(slot);
	}

	if(!make_compressed())
	{
		make_uncompressed();
	}

    const double** d_sx_N = (const double**)getVectorOfVectors(sss, n, "SpinOperation::apply_1", 's', 'x');
    const double** d_sy_N = (const double**)getVectorOfVectors(sss, n, "SpinOperation::apply_2", 's', 'y');
    const double** d_sz_N = (const double**)getVectorOfVectors(sss, n, "SpinOperation::apply_3", 's', 'z');
    const double** d_sm_N = (const double**)getVectorOfVectors(sss, n, "SpinOperation::apply_4", 's', 'm');

	      double** d_hx_N = getVectorOfVectors(sss, n, "SpinOperation::apply_5", 'h', 'x', &(slots[0]));
	      double** d_hy_N = getVectorOfVectors(sss, n, "SpinOperation::apply_6", 'h', 'y', &(slots[0]));
	      double** d_hz_N = getVectorOfVectors(sss, n, "SpinOperation::apply_7", 'h', 'z', &(slots[0]));
	
	if(compressed)
	{
		cuda_exchange_compressed_N(
			d_sx_N, d_sy_N, d_sz_N, d_sm_N,
			d_LUT, d_idx, compress_max_neighbours,
			d_hx_N, d_hy_N, d_hz_N, global_scale*0+1,
			nxyz, n);
	}
	else
	{
		cuda_exchange_N(
			d_sx_N, d_sy_N, d_sz_N, d_sm_N,
			d_strength, d_fromsite, maxFromSites,
			d_hx_N, d_hy_N, d_hz_N, global_scale*0+1,
			nx, ny, nz, n);
	}
	
	for(int i=0; i<n; i++)
	{
		sss[i]->hx[slots[i]]->new_device = true;
		sss[i]->hy[slots[i]]->new_device = true;
		sss[i]->hz[slots[i]]->new_device = true;
	}

	return true;
}

bool Exchange::apply(SpinSystem* ss)
{
	SpinSystem* sss[1];
	sss[0] = ss;
	return apply(sss, 1);
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
// 	return;
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
			size *= 2;
			size++;
			pathways = (sss*)realloc(pathways, sizeof(sss) * size);
			
			addPath(site1, site2, str);
			return;
		}
		
		pathways[num].fromsite = site1;
		pathways[num].tosite = site2;
		pathways[num].strength = str;
		num++;
		
		new_host = true;
	}
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
		lua_pushstring(L, "2 *3Vector*s, 1 Optional Number: The vectors define the lattice sites that share a pathway, the number is the strength of the pathway or 1 as a default. For example, if ex is an Exchange Operator then ex:addPath({1,1,1}, {1,1,2}, -1) and ex:addPath({1,1,2}, {1,1,1}, -1) would make two spins neighbours of each other with anti-ferromagnetic exchange.");
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
EXCHANGECUDA_API int lib_register(lua_State* L);
EXCHANGECUDA_API int lib_version(lua_State* L);
EXCHANGECUDA_API const char* lib_name(lua_State* L);
EXCHANGECUDA_API int lib_main(lua_State* L);
}

int lib_register(lua_State* L)
{
	luaT_register<Exchange>(L);
	return 0;
}

int lib_version(lua_State* L)
{
	return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Exchange-Cuda";
#else
	return "Exchange-Cuda-Debug";
#endif
}

int lib_main(lua_State* L)
{
	return 0;
}




