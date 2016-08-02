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

#include "spinoperationshortrange.h"
#include "spinsystem.h"
#include "spinsystem.hpp"
#include "info.h"
#include <strings.h>
#include <algorithm>
#include <stdlib.h>
#include <math.h>

ShortRangeCuda::ShortRangeCuda(int nx, int ny, int nz)
	: SpinOperation(ShortRangeCuda::typeName(), SHORTRANGE_SLOT, nx, ny, nz, hash32(ShortRangeCuda::typeName()))
{
	registerWS();
	d_ABoffset[0] = 0;
	h_ABoffset[0] = 0;
	d_ABvalue[0] = 0;
	h_ABvalue[0] = 0;
}

int ShortRangeCuda::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	init();
	return 0;	
}

void ShortRangeCuda::push(lua_State* L)
{
	luaT_push<ShortRangeCuda>(L, this);
}


void ShortRangeCuda::encode(buffer* b)
{
	SpinOperation::encode(b);

    for(int i=0; i<6; i++)
    {
		encodeInteger(AB[i].size(), b);
		for(unsigned int j=0; j<AB[i].size(); j++)
		{
			encodeInteger(AB[i][j].x, b);
			encodeInteger(AB[i][j].y, b);
			encodeInteger(AB[i][j].z, b);
			encodeDouble(AB[i][j].value, b);
		}
    }
}

int  ShortRangeCuda::decode(buffer* b)
{
    deinit();

	SpinOperation::decode(b);
	
    for(int i=0; i<6; i++)
    {
		AB[i].clear();
		int nn  = decodeInteger(b);
		for(int j=0; j<nn; j++)
		{
			int xx = decodeInteger(b);
			int yy = decodeInteger(b);
			int zz = decodeInteger(b);
			int vv = decodeDouble(b);

			AB[i].push_back(shortrange_data(xx,yy,zz,vv));
		}
    }

    newHostData = true;
    return 0;
}


void ShortRangeCuda::init()
{
	deinit();
	int s = nx*ny * (nz);// *2-1);
}

static int offsetOK(int nx, int ny, int nz,  int x, int y, int z, int& offset)
{
	if(x<0 || x >= nx) return 0;
	if(y<0 || y >= ny) return 0;
	if(z<0 || z >= nz) return 0;
	
	offset = x + y*nx + z*nx*ny;
	return 1;
}

double ShortRangeCuda::getAB(int matrix, int ox, int oy, int oz)
{
    ox = (ox + 10*nx)%nx;
    oy = (oy + 10*ny)%ny; //pcb
	for(unsigned int i=0; i<AB[matrix].size(); i++)
	{
		if(AB[matrix][i].x == ox && AB[matrix][i].y == oy && AB[matrix][i].z == oz)
		{
			return AB[matrix][i].value;
		}
	}
	return 0;
}

void   ShortRangeCuda::setAB(int matrix, int ox, int oy, int oz, double value)
{
    //printf("%i %i %i %i %g\n", matrix, ox, oy, oz, value);
    ox = (ox + 10*nx)%nx;
    oy = (oy + 10*ny)%ny; //pcb
	for(unsigned int i=0; i<AB[matrix].size(); i++)
	{
		if(AB[matrix][i].x == ox && AB[matrix][i].y == oy && AB[matrix][i].z == oz)
		{
			AB[matrix][i].value = value;
			return;
		}
	}
	//printf("PB(%i)\n", AB[matrix].size());
	AB[matrix].push_back(shortrange_data(ox,oy,oz,value));
	newHostData = true;
}

	
#define getsetPattern(AB,m) \
double ShortRangeCuda::get##AB (int ox, int oy, int oz) \
{ \
	return getAB(m, ox,oy,oz);\
} \
void  ShortRangeCuda::set##AB (int ox, int oy, int oz, double value) \
{ \
	return setAB(m, ox,oy,oz,value); \
}

getsetPattern(XX,0)
getsetPattern(XY,1)
getsetPattern(XZ,2)
getsetPattern(YY,3)
getsetPattern(YZ,4)
getsetPattern(ZZ,5)

static bool shortrange_sort(const shortrange_data& i, const shortrange_data& j)
{
	if(i.z < j.z) return true;
	if(i.y < j.y) return true;
	if(i.x < j.x) return true;
	return false; 
}

void ShortRangeCuda::compile()
{
	if(!newHostData) return;

// 	vector< vector<shortrange_data>* > vv;
// 	
// 	vv.push_back(&XX);
// 	vv.push_back(&XY);
// 	vv.push_back(&XZ);
// 
// 	vv.push_back(&YY);
// 	vv.push_back(&YZ);
// 	vv.push_back(&ZZ);
	
	if(d_ABoffset[0])
	{
		for(int i=0; i<6; i++)
		{
			free_device(d_ABoffset[i]);
			free_host(h_ABoffset[i]);
		}
	}
	if(d_ABvalue[0])
	{
		for(int i=0; i<6; i++)
		{
			free_device(d_ABvalue[i]);
			free_host(h_ABvalue[i]);
		}
	}
	
	//first sort memory accesses
	for(int i=0; i<6; i++)
	{
		vector<shortrange_data>& v = AB[i];
		sort(v.begin(), v.end(), shortrange_sort);
	}
	
	//find number with finite values
	for(int i=0; i<6; i++)
	{
		ABcount[i] = 0;
		vector<shortrange_data>& v = AB[i];
		for(unsigned int j=0; j<v.size(); j++)
		{
			if(v[j].value != 0)
			{
				ABcount[i]++;
			}
		}
// 		printf("CC %i %i  %i\n", i, ABcount[i], v.size());
	}
	
	//allocate host/device memory
	for(int i=0; i<6; i++)
	{
		const int j = ABcount[i];
		malloc_device( &(d_ABvalue[i]),  sizeof(float)*j+1); //+1 to take care of zero case
		malloc_host  ( &(h_ABvalue[i]),  sizeof(float)*j+1);
		
		malloc_device( &(d_ABoffset[i]), sizeof(int)*j*3+1);
		malloc_host  ( &(h_ABoffset[i]), sizeof(int)*j*3+1);
	}
	
	//populate host memory
	for(int i=0; i<6; i++)
	{
		int k = 0;
		vector<shortrange_data>& v = AB[i];
		for(unsigned int j=0; j<v.size(); j++)
		{
			if(v[j].value != 0)
			{
// 				printf("M%i) %i %i %i  %g\n", i, v[j].x, v[j].y, v[j].z, v[j].value);
				h_ABoffset[i][k*3+0] = v[j].x;
				h_ABoffset[i][k*3+1] = v[j].y;
				h_ABoffset[i][k*3+2] = v[j].z;
				h_ABvalue[i][k] = v[j].value;
				k++;
			}
		}
	}
	
	//transfer to device
	for(int i=0; i<6; i++)
	{
		const int j = ABcount[i];
		memcpy_h2d(d_ABoffset[i], h_ABoffset[i], j*sizeof(int)*3);
		memcpy_h2d(d_ABvalue[i],  h_ABvalue[i],  j*sizeof(float));
	}
	
	

	newHostData = false;
}

	
void ShortRangeCuda::deinit()
{
	if(d_ABoffset[0])
	{
		for(int i=0; i<6; i++)
		{
			free_device(d_ABoffset[i]);
			free_host(h_ABoffset[i]);
		}
	}
	if(d_ABvalue[0])
	{
		for(int i=0; i<6; i++)
		{
			free_device(d_ABvalue[i]);
			free_host(h_ABvalue[i]);
		}
	}
	d_ABoffset[0] = 0;
	d_ABvalue[0] = 0;
}

ShortRangeCuda::~ShortRangeCuda()
{
	unregisterWS();
	deinit();
}



bool ShortRangeCuda::applyToSum(SpinSystem* ss)
{
	if(newHostData)
		compile();

	ss->sync_spins_hd();
	ss->ensureSlotExists(slot);

	int field_ws = sizeof(float)*nx*ny*nz;
	
	float* d_wsx;
	float* d_wsy;
	float* d_wsz;
	
	getWSMem(&d_wsx,  field_ws,
			 &d_wsy,  field_ws,
			 &d_wsz,  field_ws);
		  
	const float* d_sx = ss->d_x;
	const float* d_sy = ss->d_y;
	const float* d_sz = ss->d_z;
	
	JM_SHORTRANGE(nx, ny, nz, 
				    global_scale,
					ABcount, d_ABoffset, d_ABvalue,
					d_sx, d_sy, d_sz, 
					d_wsx, d_wsy, d_wsz);

	const int nxyz = nx*ny*nz;
	cuda_addArrays32(ss->d_hx[SUM_SLOT], nxyz, ss->d_hx[SUM_SLOT], d_wsx);
	cuda_addArrays32(ss->d_hy[SUM_SLOT], nxyz, ss->d_hy[SUM_SLOT], d_wsy);
	cuda_addArrays32(ss->d_hz[SUM_SLOT], nxyz, ss->d_hz[SUM_SLOT], d_wsz);
	ss->slot_used[SUM_SLOT] = true;
	
	return true;
}

bool ShortRangeCuda::apply(SpinSystem* ss)
{
	if(newHostData)
		compile();

	markSlotUsed(ss);
	ss->ensureSlotExists(slot);

	ss->sync_spins_hd();

	float* d_hx = ss->d_hx[slot];
	float* d_hy = ss->d_hy[slot];
	float* d_hz = ss->d_hz[slot];

	const float* d_sx = ss->d_x;
	const float* d_sy = ss->d_y;
	const float* d_sz = ss->d_z;
	
	
 	JM_SHORTRANGE(nx, ny, nz, 
  				  global_scale,
				  ABcount, d_ABoffset, d_ABvalue,
				  d_sx, d_sy, d_sz, 
				  d_hx, d_hy, d_hz);
	
	ss->new_device_fields[slot] = true;
	return true;
}








int l_set(lua_State* L)
{
	LUA_PREAMBLE(ShortRangeCuda, sr, 1);
	const char* badname = "1st argument must be matrix name: XX, XY, XZ, YY, YZ or ZZ";
	
	if(!lua_isstring(L, 2))
	    return luaL_error(L, badname);

	const char* type = lua_tostring(L, 2);

	const char* names[6] = {"XX", "XY", "XZ", "YY", "YZ", "ZZ"};
	int mat = -1;
	for(int i=0; i<6; i++)
	{
	    if(strcasecmp(type, names[i]) == 0)
	    {
		mat = i;
	    }
	}

	if(mat < 0)
	    return luaL_error(L, badname);


	int offset[3];

	int r1 = lua_getNint(L, 3, offset, 3, 0);
        if(r1<0)
	    return luaL_error(L, "invalid offset");

	double val = lua_tonumber(L, 3+r1);

	// not altering zero base here:
	sr->setAB(mat, offset[0], offset[1], offset[2], val);

	return 0;
}

int l_get(lua_State* L)
{
	LUA_PREAMBLE(ShortRangeCuda, sr, 1);
	const char* badname = "1st argument must be matrix name: XX, XY, XZ, YY, YZ or ZZ";
	
	if(!lua_isstring(L, 2))
	    return luaL_error(L, badname);

	const char* type = lua_tostring(L, 2);

	const char* names[6] = {"XX", "XY", "XZ", "YY", "YZ", "ZZ"};
	int mat = -1;
	for(int i=0; i<6; i++)
	{
	    if(strcasecmp(type, names[i]) == 0)
	    {
		mat = i;
	    }
	}

	if(mat < 0)
	    return luaL_error(L, badname);

	int offset[3];

	int r1 = lua_getNint(L, 3, offset, 3, 0);
        if(r1<0)
	    return luaL_error(L, "invalid offset");

	// not altering zero base here:
	double val = sr->getAB(mat, offset[0], offset[1], offset[2]);

	lua_pushnumber(L, val);
	return 1;
}


int ShortRangeCuda::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates a custom Range Field of a *SpinSystem*");
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
	
	if(func == l_get)
	{
		lua_pushstring(L, "Get an element of an interaction matrix");
		lua_pushstring(L, "1 string, 1 *3Vector*: The string indicates which AB matrix to access. Can be XX, XY, XZ, YY, YZ or ZZ. The *3Vector* indexes into the matrix. Note: indexes are zero-based and are interpreted as offsets.");
		lua_pushstring(L, "1 number: The fetched value.");
		return 3;
	}

	if(func == l_set)
	{
		lua_pushstring(L, "Set an element of an interaction matrix");
		lua_pushstring(L, "1 string, 1 *3Vector*, 1 number: The string indicates which AB matrix to access. Can be XX, XY, XZ, YY, YZ or ZZ. The *3Vector* indexes into the matrix. The number is the value that is set at the index. Note: indexes are zero-based and are interpreted as offsets.");
		lua_pushstring(L, "");
		return 3;
	}
	
	return 0;
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* ShortRangeCuda::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, SpinOperation::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"get",      l_get},
		{"set",      l_set},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}


extern "C"
{
SHORTRANGECUDA_API int lib_register(lua_State* L);
SHORTRANGECUDA_API int lib_version(lua_State* L);
SHORTRANGECUDA_API const char* lib_name(lua_State* L);
SHORTRANGECUDA_API int lib_main(lua_State* L);
}

SHORTRANGECUDA_API int lib_register(lua_State* L)
{
	luaT_register<ShortRangeCuda>(L);
	return 0;
}


SHORTRANGECUDA_API int lib_version(lua_State* L)
{
	return __revi;
}


SHORTRANGECUDA_API const char* lib_name(lua_State* L)
{
	return "ShortRange-Cuda";
}

SHORTRANGECUDA_API int lib_main(lua_State* L)
{
	return 0;
}


