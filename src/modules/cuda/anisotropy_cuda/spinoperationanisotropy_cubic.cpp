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

#include "spinoperationanisotropy_cubic.h"
#include "spinoperationanisotropy_cubic.hpp"
#include "spinsystem.h"

#include <vector>
#include <algorithm>
#include <stdlib.h>

// 3 axes and 3 K constants
#define LUT_ENTRY_LENGTH (3+3+3+3)

AnisotropyCubic::AnisotropyCubic(int nx, int ny, int nz)
	: SpinOperation(nx, ny, nz, hash32(AnisotropyCubic::typeName()))
{
	d_nx[0] = 0;
	d_LUT = 0;
	d_idx = 0;
	
	new_host = true;
	compressed = false;
	newDataFromScript = false;

	ops = 0;
	//size = nx*ny*nz;
	size = 0;
	init();
}

const char* AnisotropyCubic::getSlotName()
{
	return "CubicAnisotropy";
}


int AnisotropyCubic::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	size = 0;
// 	size = nx*ny*nz;
	init();
	return 0;
}

void AnisotropyCubic::init()
{
	num = 0;
	if(size < 0)
		size = 1;

	ops = (ani*)malloc(sizeof(ani) * size);
	d_nx[0] = 0;
}

void AnisotropyCubic::deinit()
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



void AnisotropyCubic::delete_host()
{
	if(d_nx[0])
	{
		for(int i=0; i<3; i++)
		{
			free_host(h_nx[i]);
			free_host(h_ny[i]);
			free_host(h_nz[i]);
			free_host(h_k[i]);
		}

		d_nx[0] = 0;
	}	
}
bool AnisotropyCubic::make_host()
{
	if(h_nx[0])
		return true;

	for(int i=0; i<3; i++)
	{
		malloc_host(&(h_nx[i]), sizeof(double) * nxyz);
		malloc_host(&(h_ny[i]), sizeof(double) * nxyz);
		malloc_host(&(h_nz[i]), sizeof(double) * nxyz);
		malloc_host(&(h_k[i]),  sizeof(double) * nxyz);
	}

	for(int j=0; j<3; j++)
	for(int i=0; i<nxyz; i++)
	{
		h_nx[j][i] = 0;
		h_ny[j][i] = 0;
		h_nz[j][i] = 0;
		h_k[j][i]  = 0;
	}
	return true;
}

void AnisotropyCubic::delete_compressed()
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

void AnisotropyCubic::delete_uncompressed()
{
	void*** a[4] = {
		(void***)&d_nx, (void***)&d_ny,
		(void***)&d_nz, (void***)&d_k
	};
	
	for(int i=0; i<4; i++)
	{
		for(int j=0; j<3; j++)
		{
			if(*a[i][j])
				free_device(*a[i][j]);
			*a[i][j] = 0;
		}
	}
}

class sani3
{
public:
	sani3(int s, double* a1, double* a2, double* a3, double* _K)
	{
		site = s;
		memcpy(axis[0], a1, sizeof(double)*3);
		memcpy(axis[1], a2, sizeof(double)*3);
		memcpy(axis[2], a3, sizeof(double)*3);
		memcpy(k, _K, sizeof(double)*3);
		
		// making axes +x
		for(int i=0; i<2; i++)
		{
			if(axis[i][0] < 0)
			{
				axis[i][0] *= -1.0;
				axis[i][1] *= -1.0;
				axis[i][2] *= -1.0;
			}
		}
	}
	sani3(const sani3& s)
	{
		site = s.site;
		memcpy(axis[0], s.axis[0], sizeof(double)*3);
		memcpy(axis[1], s.axis[1], sizeof(double)*3);
		memcpy(axis[2], s.axis[3], sizeof(double)*3);
		memcpy(k, s.k, sizeof(double)*3);
	
	}

	// same other than site
	bool operator==(const sani3 &other) const
	{
		for(int i=0; i<3; i++)
		{
			if(axis[0][i] != other.axis[0][i]) return false;
			if(axis[1][i] != other.axis[1][i]) return false;
			if(axis[2][i] != other.axis[2][i]) return false;
			if(k[i] != other.k[i]) return false;
		}
		return true;
	}
	
	// implementation doesn't really matter. Just needs to be consistent
	bool operator<(const sani3 &other) const
	{
		for(int i=0; i<3; i++)
		{
			if(axis[0][i] >= other.axis[0][i]) return false;
			if(axis[1][i] >= other.axis[1][i]) return false;
			if(axis[2][i] >= other.axis[2][i]) return false;
			if(k[i] >= other.k[i]) return false;
		}
		return true;
	}
	
	int site;
	double axis[3][3];
	double k[3];
	char id;
};

bool AnisotropyCubic::make_compressed()
{
	if(compressAttempted)
		return compressed;
		
	delete_compressed();
	delete_uncompressed();
	
	compressAttempted = true;
	if(!nxyz)
		return false;
	

	
	compressing = true;
	
	vector<sani3> aa;
	for(int i=0; i<nxyz; i++)
	{
		double a1[3];
		double a2[3];
		double a3[3];
		double k[3];
		
		a1[0] = h_nx[0][i];
		a1[1] = h_ny[0][i];
		a1[2] = h_nz[0][i];

		a2[0] = h_nx[1][i];
		a2[1] = h_ny[1][i];
		a2[2] = h_nz[1][i];
		
		a3[0] = h_nx[2][i];
		a3[1] = h_ny[2][i];
		a3[2] = h_nz[2][i];
		
		k[0] = h_k[0][i];
		k[1] = h_k[1][i];
		k[2] = h_k[2][i];
		
		aa.push_back(sani3(i, a1, a2, a3, k));
	}
		
	
	std::sort(aa.begin(), aa.end());
	
	unsigned int last_one = 0;
	
	vector<unsigned int> uu; //uniques
	uu.push_back(0);
	aa[0].id = 0;
	
	for(int i=1; i<nxyz; i++)
	{
		if(!(aa[i] ==  aa[last_one]))
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
	
	malloc_host(&h_LUT, sizeof(double) * unique * LUT_ENTRY_LENGTH);
	malloc_host(&h_idx, sizeof(unsigned char) * nxyz);
	
	for(unsigned int i=0; i<uu.size(); i++)
	{
		sani3& q = aa[ uu[i] ];
		h_LUT[i*LUT_ENTRY_LENGTH+0] = q.axis[0][0];
		h_LUT[i*LUT_ENTRY_LENGTH+1] = q.axis[0][1];
		h_LUT[i*LUT_ENTRY_LENGTH+2] = q.axis[0][2];

		h_LUT[i*LUT_ENTRY_LENGTH+3] = q.axis[1][0];
		h_LUT[i*LUT_ENTRY_LENGTH+4] = q.axis[1][1];
		h_LUT[i*LUT_ENTRY_LENGTH+5] = q.axis[1][2];

		h_LUT[i*LUT_ENTRY_LENGTH+6] = q.axis[2][0];
		h_LUT[i*LUT_ENTRY_LENGTH+7] = q.axis[2][1];
		h_LUT[i*LUT_ENTRY_LENGTH+8] = q.axis[2][2];

		h_LUT[i*LUT_ENTRY_LENGTH+ 9] = q.k[0];
		h_LUT[i*LUT_ENTRY_LENGTH+10] = q.k[1];
		h_LUT[i*LUT_ENTRY_LENGTH+11] = q.k[2];
	}
	
	for(int i=0; i<nxyz; i++)
	{
		h_idx[i] = aa[i].id;
	}
	
	bool ok;
	ok  = malloc_device(&d_LUT, sizeof(double) * unique * LUT_ENTRY_LENGTH);
	ok &= malloc_device(&d_idx, sizeof(unsigned char) * nxyz);
	
	if(!ok)
	{
		delete_compressed(); //incase LUT generated
		compressed = false;
		compressing = false;
		//should probably say something: this is bad.
		return false;
	}
	
	memcpy_h2d(d_LUT, h_LUT, sizeof(double) * unique * LUT_ENTRY_LENGTH);
	memcpy_h2d(d_idx, h_idx, sizeof(unsigned char) * nxyz);
	
	free_host(h_LUT);
	free_host(h_idx);
	
	compressed = true;
	compressing = false;
	return true;
}



#include <algorithm>    // std::sort
#include <vector>       // std::vector
using namespace std;
// this is messy but more efficient than before
int AnisotropyCubic::merge()
{
	fprintf(stderr, "AnisotropyCubic merge() is unimplemented\n");
	return 0;
#if 0
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
#endif
}

bool AnisotropyCubic::getAnisotropy(int site, double* a1, double* a2, double* a3, double* K3)
{
	for(int i=0; i<num; i++)
	{
		if(ops[i].site == site)
		{
			memcpy(a1, ops[i].axis[0], sizeof(double)*3);
			memcpy(a2, ops[i].axis[1], sizeof(double)*3);
			memcpy(a3, ops[i].axis[2], sizeof(double)*3);
			memcpy(K3, ops[i].K,       sizeof(double)*3);
			return true;
		}
	}
	
	a1[0] = 1; a1[1] = 0; a1[2] = 0;
	a2[0] = 0; a2[1] = 1; a2[2] = 0;
	a3[0] = 0; a3[1] = 0; a3[2] = 1;
	
	K3[0] = 0;
	K3[1] = 0;
	K3[2] = 0;
	
	return false;
}

static void cross(double* a, double* b, double* c)
{
	c[0] = a[1]*b[2] - a[2]*b[1];
	c[1] = a[2]*b[0] - a[0]*b[2];
	c[2] = a[0]*b[1] - a[1]*b[0];
}
static double dot(double* a, double* b)
{
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static void scale(double s, double* a)
{
	a[0] *= s;
	a[1] *= s;
	a[2] *= s;
}

void AnisotropyCubic::addAnisotropy(int site, double* a1, double* a2, double* K3)
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
	memcpy(ops[num].axis[0], a1, sizeof(double)*3);
	memcpy(ops[num].axis[1], a2, sizeof(double)*3);
	
	cross(ops[num].axis[0], ops[num].axis[1], ops[num].axis[2]);

	for(int i=0; i<3; i++)
	{
		const double length = dot(ops[num].axis[i], ops[num].axis[i]);
		if(length > 0)
			scale(1.0/length, ops[num].axis[i]);
	}
	
	memcpy(ops[num].K, K3, sizeof(double)*3);
	num++;
}

void AnisotropyCubic::encode(buffer* b)
{
	SpinOperation::encode(b); //nx,ny,nz,global_scale

	char version = 0;
	encodeChar(version, b);
	
	encodeInteger(num, b);
	for(int i=0; i<num; i++)
	{
		encodeInteger(ops[i].site, b);
		for(int axis=0; axis<2; axis++) // JUST ENCODING 2 AXIS, 3RD IS COMPUTED
			for(int coordinate=0; coordinate<3; coordinate++)
				encodeDouble(ops[i].axis[axis][coordinate], b);

		for(int j=0; j<3; j++)
			encodeDouble(ops[i].K[j], b);
	}
}

int AnisotropyCubic::decode(buffer* b)
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
			
			double a[2][3];
			double K[3];
			
			for(int axis=0; axis<2; axis++) // JUST DECODING 2 AXIS, 3RD IS COMPUTED
				for(int coordinate=0; coordinate<3; coordinate++)
					a[axis][coordinate] = decodeDouble(b);
			
			for(int j=0; j<3; j++)
				K[j] = decodeDouble(b);

			addAnisotropy(site, a[0], a[1], K);
		}
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}
	return 0;
}


AnisotropyCubic::~AnisotropyCubic()
{
	deinit();
}


bool AnisotropyCubic::apply(SpinSystem* ss)
{
	SpinSystem* sss[1];
	sss[0] = ss;
	return apply(sss, 1);
}

// convert from cpu style info to gpu precursor style info
void AnisotropyCubic::writeToMemory()
{
	if(!newDataFromScript)
		return;
	newDataFromScript = false;

	make_uncompressed();
// 	merge();

	for(int i=0; i<num; i++)
	{
		const int site = ops[i].site;
		double nx1 = ops[i].axis[0][0];
		double nx2 = ops[i].axis[1][0];
		double nx3 = ops[i].axis[2][0];

		double ny1 = ops[i].axis[0][1];
		double ny2 = ops[i].axis[1][1];
		double ny3 = ops[i].axis[2][1];

		double nz1 = ops[i].axis[0][2];
		double nz2 = ops[i].axis[1][2];
		double nz3 = ops[i].axis[2][2];

		double K1 = ops[i].K[0];
		double K2 = ops[i].K[1];
		double K3 = ops[i].K[2];

		if(site >= 0 && site < nxyz)
		{
			make_host();
		
			h_nx[0][site] = nx1;
			h_nx[1][site] = nx2;
			h_nx[2][site] = nx3;

			h_ny[0][site] = ny1;
			h_ny[1][site] = ny2;
			h_ny[2][site] = ny3;

			h_nz[0][site] = nz1;
			h_nz[1][site] = nz2;
			h_nz[2][site] = nz3;

			h_k[0][site] = K1;
			h_k[1][site] = K2;
			h_k[2][site] = K3;
			
			new_host = true;
			compressAttempted = false;
			delete_compressed();
			delete_uncompressed();
		}
	}
}


// E_anis = K1 * (<axis1,m>^2 <axis2,m>^2 + <axis1,m>^2 <axis3,m>^2 + <axis2,m>^2 <axis3,m>^2)
//        + K2 * (<axis1,m>^2 <axis2,m>^2 <axis3,m>^2)
//        + K3 * (<axis1,m>^4 <axis2,m>^4 + <axis1,m>^4 <axis3,m>^4 + <axis2,m>^4 <axis3,m>^4)

bool AnisotropyCubic::apply(SpinSystem** sss, int n)
{
	vector<int> slots;
	for(int i=0; i<n; i++)
		slots.push_back(markSlotUsed(sss[i]));

	writeToMemory();
	if(!make_compressed())
		make_uncompressed();
	
	const double** d_sx_N = (const double**)SpinOperation::getVectorOfVectors(sss, n, "SpinOperation::apply_1", 's', 'x');
	const double** d_sy_N = (const double**)SpinOperation::getVectorOfVectors(sss, n, "SpinOperation::apply_2", 's', 'y');
	const double** d_sz_N = (const double**)SpinOperation::getVectorOfVectors(sss, n, "SpinOperation::apply_3", 's', 'z');

	      double** d_hx_N = SpinOperation::getVectorOfVectors(sss, n, "SpinOperation::apply_4", 'h', 'x', &(slots[0]));
	      double** d_hy_N = SpinOperation::getVectorOfVectors(sss, n, "SpinOperation::apply_5", 'h', 'y', &(slots[0]));
	      double** d_hz_N = SpinOperation::getVectorOfVectors(sss, n, "SpinOperation::apply_6", 'h', 'z', &(slots[0]));
	
	if(compressed)
	{
		// d_LUT is non-null (since compressed)
		cuda_anisotropy_cubic_compressed_N(
			global_scale,
			d_sx_N, d_sy_N, d_sz_N,
			d_LUT, d_idx, 
			d_hx_N, d_hy_N, d_hz_N,
			nxyz, n);
	}
	else
	{
		if(!d_nx[0])
			make_uncompressed();
		
		cuda_anisotropy_cubic_N(
			global_scale,
			d_sx_N, d_sy_N, d_sz_N,
			d_nx[0], d_nx[1], d_nx[2],
			d_ny[0], d_ny[1], d_ny[2],
			d_nz[0], d_nz[1], d_nz[2],
			d_k[0], d_k[1], d_k[2],
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
	LUA_PREAMBLE(AnisotropyCubic, ani, 1);

	double a[3][3];
	double K[3];

	int p[3];
	int r1 = lua_getNint(L, 3, p, 2, 1);

	if(r1<0)
		return luaL_error(L, "invalid site format");
	
	if(!ani->member(p[0]-1, p[1]-1, p[2]-1))
		return luaL_error(L, "site is not part of system");

	int idx = ani->getidx(p[0]-1, p[1]-1, p[2]-1);
	

	ani->getAnisotropy(idx, a[0],a[1],a[2],K);
	
	for(int i=0; i<3; i++)
	{
		lua_newtable(L);
		lua_pushinteger(L, 1); lua_pushnumber(L, a[i][0]); lua_settable(L, -3);
		lua_pushinteger(L, 2); lua_pushnumber(L, a[i][1]); lua_settable(L, -3);
		lua_pushinteger(L, 3); lua_pushnumber(L, a[i][2]); lua_settable(L, -3);
	}
	lua_pushnumber(L, K[0]);
	lua_pushnumber(L, K[1]);
	lua_pushnumber(L, K[2]);

	return 6;
}



static int l_numofax(lua_State* L)
{
	LUA_PREAMBLE(AnisotropyCubic, ani, 1);
	lua_pushinteger(L, ani->num);
	return 1;
}


static int l_axisat(lua_State* L)
{
	LUA_PREAMBLE(AnisotropyCubic, ani, 1);
	
	int idx = lua_tointeger(L, 2) - 1;

	if(idx < 0 || idx >= ani->num)
		return luaL_error(L, "Invalid axis index");
	

	const int site = ani->ops[idx].site;
	double* K = ani->ops[idx].K;
	
	int x,y,z;
	ani->idx2xyz(site, x, y, z);

	lua_newtable(L);
	lua_pushinteger(L, 1); lua_pushinteger(L, x+1); lua_settable(L, -3);
	lua_pushinteger(L, 2); lua_pushinteger(L, y+1); lua_settable(L, -3);
	lua_pushinteger(L, 3); lua_pushinteger(L, z+1); lua_settable(L, -3);
	
	for(int i=0; i<3; i++)
	{
		lua_newtable(L);
		for(int j=0; j<3; j++)
		{
			lua_pushinteger(L, j+1);
			lua_pushnumber(L, ani->ops[idx].axis[i][j]);
			lua_settable(L, -3);
		}
	}
	
	for(int i=0; i<3; i++)
		lua_pushnumber(L, K[i]);
	
	return 7;
}

static int l_add(lua_State* L)
{
	LUA_PREAMBLE(AnisotropyCubic, ani, 1);

	int p[3];

	int r1 = lua_getNint(L, 3, p, 2, 1);
	
	if(r1<0)
		return luaL_error(L, "invalid site format");
	
	if(!ani->member(p[0]-1, p[1]-1, p[2]-1))
		return luaL_error(L, "site (%d, %d, %d) is not part of operator (%dx%dx%d)", p[0], p[1], p[2], ani->nx, ani->ny, ani->nz);

	int idx = ani->getidx(p[0]-1, p[1]-1, p[2]-1);

	double a[2][3];	
	int r2 = lua_getNdouble(L, 3, a[0], 2+r1, 0);
	if(r2<0)
		return luaL_error(L, "invalid anisotropy axis");
	int r3 = lua_getNdouble(L, 3, a[1], 2+r1+r2, 0);
	if(r3<0)
		return luaL_error(L, "invalid anisotropy axis");


	int t = 0;
	double K[3] = {0,0,0};
	for(int i=2+r1+r2+r3; i<=lua_gettop(L) && t < 3; i++)
	{
		if(lua_isnumber(L, i))
		{
			K[t] = lua_tonumber(L, i);
			t++;
		}
	}

	if(t == 0)
		return luaL_error(L, "anisotropy needs strength");
	
	ani->addAnisotropy(idx, a[0], a[1], K);
	return 0;
}

static int l_mergeAxes(lua_State* L)
{
	LUA_PREAMBLE(AnisotropyCubic, ani, 1);
	lua_pushinteger(L, ani->merge());
	return 1;	
}

int AnisotropyCubic::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Computes the cubic anisotropy fields for a *SpinSystem* as the derivative of the following energy expression.\n"
					"<pre>"
					"E_anis = K1 * (&lt;axis1,m&gt;^2 &lt;axis2,m&gt;^2 + &lt;axis1,m&gt;^2 &lt;axis3,m&gt;^2 + &lt;axis2,m&gt;^2 &lt;axis3,m&gt;^2)\n"
					"       + K2 * (&lt;axis1,m&gt;^2 &lt;axis2,m&gt;^2 &lt;axis3,m&gt;^2)\n"
					"       + K3 * (&lt;axis1,m&gt;^4 &lt;axis2,m&gt;^4 + &lt;axis1,m&gt;^4 &lt;axis3,m&gt;^4 + &lt;axis2,m&gt;^4 &lt;axis3,m&gt;^4)\n"
					"</pre>"
					"K1 = Fourth-order phenomenological anisotropy constant\n"
					"K2 = Sixth-order phenomenological anisotropy constant\n"
					"K3 = Eigth-order phenomenological anisotropy constant</pre>"
		);
		
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
		
	if(func == l_add)
	{
		lua_pushstring(L, "Add a lattice site to the anisotropy calculation");
		lua_pushstring(L, "3 *3Vector*s, 1 to 3 numbers: The first *3Vector* defines a lattice site, the second and third define an anisotropy axes which will be made unit vectors and crossed to create the 3rd cubic axis.  The numbers define the 4th, 6th and 8th order phenomenological constants, the first is required and the last two have default values of zero.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_get)
	{
		lua_pushstring(L, "Fetch the anisotropy direction and magnitude at a given site.");
		lua_pushstring(L, "1 *3Vector*: The *3Vector* defines a lattice site.");
		lua_pushstring(L, "3 Tables of 3 Nubmers, 3 Numbers: The tables define the axes and the numbers are the 4th, 6th and 8th order phenomenological constants.");
		return 3;
	}
	
	if(func == l_axisat)
	{
		lua_pushstring(L, "Return the site, easy axis and strength at the given index.");
		lua_pushstring(L, "1 Integer: Index of the axis.");
		lua_pushstring(L, "1 Table of 3 Integers, 3 Tables of 3 Nubmers, 3 Numbers: Coordinates of the site, directions of the cubic axes and values of the 4th, 6th and 8th order phenomenological constants.");
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
const luaL_Reg* AnisotropyCubic::luaMethods()
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



