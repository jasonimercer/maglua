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

#include "spinsystem.h"
#include "info.h"
#include "spinoperationlongrange2d.h"
#include "luamigrate.h"

#include <stdlib.h>
#include <math.h>

#if defined NDEBUG || defined __OPTIMIZE__
#define DDD
#else
#define DDD printf("(%s:%i)\n", __FILE__, __LINE__);
#endif

LongRange2D::LongRange2D(int nx, int ny, int nz, const int encode_tag)
	: SpinOperation(nx, ny, nz, encode_tag)
{
	registerWS();
    qXX = 0;
    XX = 0;
	ws1 = 0;
	g = 0;

	longrange_ref = LUA_REFNIL;
	function_ref = LUA_REFNIL;

	XX = 0;

	compileRequired = true;
	newDataRequired = true;
}
int LongRange2D::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	LongRange2D::init();
	return 0;	
}


void LongRange2D::encode(buffer* b)
{
	SpinOperation::encode(b);
	char version = 0;
	encodeChar(version, b);
	
	
	for(int i=0; i<nz; i++)
	{
		encodeDouble(g[i], b);
	}
	
	int ref[2];
	ref[0] = longrange_ref;
	ref[1] = function_ref;

	for(int i=0; i<2; i++)
	{
		if(ref[i] != LUA_REFNIL)
			lua_rawgeti(L, LUA_REGISTRYINDEX, ref[i]);
		else
			lua_pushnil(L);

		_exportLuaVariable(L, lua_gettop(L), b);
		lua_pop(L, 1);
	}
}

int  LongRange2D::decode(buffer* b)
{
	SpinOperation::decode(b);
	char version = decodeChar(b);
	if(version == 0)
	{
		if(g)
			delete [] g;
		g = new double[nz];
		
		for(int i=0; i<nz; i++)
		{
			g[i] = decodeDouble(b);
		}
		
		int n = lua_gettop(L);


		if(longrange_ref != LUA_REFNIL)
			luaL_unref(L, LUA_REGISTRYINDEX, longrange_ref);
		if(function_ref != LUA_REFNIL)
			luaL_unref(L, LUA_REGISTRYINDEX, function_ref);

		_importLuaVariable(L, b);
		longrange_ref = luaL_ref(L, LUA_REGISTRYINDEX);

		_importLuaVariable(L, b);
		function_ref = luaL_ref(L, LUA_REGISTRYINDEX);

		while(lua_gettop(L) > n)
			lua_pop(L, 1);
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}

	return 0;
}

dArray* LongRange2D::getLAB(int layer_dest, int layer_src, const char* AB)
{
	if(layer_dest >= 0 || layer_dest < nz)
		if(layer_src >= 0 || layer_src < nz)
		{
			if(strncasecmp(AB, "xx", 2) == 0)
				return XX[layer_dest][layer_src];
			if(strncasecmp(AB, "xy", 2) == 0)
				return XY[layer_dest][layer_src];
			if(strncasecmp(AB, "xz", 2) == 0)
				return XZ[layer_dest][layer_src];

			if(strncasecmp(AB, "yx", 2) == 0)
				return YX[layer_dest][layer_src];
			if(strncasecmp(AB, "yy", 2) == 0)
				return YY[layer_dest][layer_src];
			if(strncasecmp(AB, "yz", 2) == 0)
				return YZ[layer_dest][layer_src];
			
			if(strncasecmp(AB, "zx", 2) == 0)
				return ZX[layer_dest][layer_src];
			if(strncasecmp(AB, "zy", 2) == 0)
				return ZY[layer_dest][layer_src];
			if(strncasecmp(AB, "zz", 2) == 0)
				return ZZ[layer_dest][layer_src];
		}
	return 0;
}

void LongRange2D::setLAB(int layer_dest, int layer_src, const char* AB, dArray* newArray)
{
	dArray*** a[9];
	a[0] = XX;	a[1] = XY;	a[2] = XZ;
	a[3] = YX;	a[4] = YY;	a[5] = YZ;
	a[6] = ZX;	a[7] = ZY;	a[8] = ZZ;
	
	const char* n[9] = {
		"XX", "XY", "XZ",
		"YX", "YY", "YZ",
		"ZX", "ZY", "ZZ"};
		
	luaT_inc<dArray>(newArray);
	if(layer_dest >= 0 || layer_dest < nz)
		if(layer_src >= 0 || layer_src < nz)
		{
			for(int i=0; i<9; i++)
			{
				if(strncasecmp(AB, n[i], 2) == 0)
				{
					luaT_dec<dArray>(a[i][layer_dest][layer_src]);
					a[i][layer_dest][layer_src] = newArray;
				}
			}
		}
}


template <typename T>
static T*** initAB(const int nx, const int ny, const int nz)
{
	T*** a = new T** [nz];
	for(int i=0; i<nz; i++)
	{
		a[i] = new T* [nz];
		for(int j=0; j<nz; j++)
		{
			a[i][j] = luaT_inc<T>(new T(nx,ny,1));
			a[i][j]->zero();
		}
	}
	return a;
}

void LongRange2D::init()
{
	if(XX) return;

	deinit();

	g = new double[nz];
	for(int i=0; i<nz; i++)
	{
		g[i] = 1;
	}
	
	hqx = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	hqy = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	hqz = luaT_inc<dcArray>(new dcArray(nx,ny,nz));

	hrx = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	hry = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	hrz = luaT_inc<dcArray>(new dcArray(nx,ny,nz));

	qXX = initAB<dcArray>(nx,ny,nz);
	qXY = initAB<dcArray>(nx,ny,nz);
	qXZ = initAB<dcArray>(nx,ny,nz);
	
	qYX = initAB<dcArray>(nx,ny,nz);
	qYY = initAB<dcArray>(nx,ny,nz);
	qYZ = initAB<dcArray>(nx,ny,nz);
	
	qZX = initAB<dcArray>(nx,ny,nz);
	qZY = initAB<dcArray>(nx,ny,nz);
	qZZ = initAB<dcArray>(nx,ny,nz);
	
	XX = initAB<dArray>(nx,ny,nz);
	XY = initAB<dArray>(nx,ny,nz);
	XZ = initAB<dArray>(nx,ny,nz);
	
	YX = initAB<dArray>(nx,ny,nz);
	YY = initAB<dArray>(nx,ny,nz);
	YZ = initAB<dArray>(nx,ny,nz);

	ZX = initAB<dArray>(nx,ny,nz);
	ZY = initAB<dArray>(nx,ny,nz);
	ZZ = initAB<dArray>(nx,ny,nz);
	
	// these tags are the same as lr3d but we're assuming that we can't
	// apply 2 operators simultaneously (this is a safe assumption)
	ws1 = getWSdcArray(nx,ny,nz, hash32("SpinOperation::apply_1"));
	ws2 = getWSdcArray(nx,ny,nz, hash32("SpinOperation::apply_2"));
}

template <typename T>
static void decAB(T*** a, int nz)
{
	for(int i=0; i<nz; i++)
	{
		for(int j=0; j<nz; j++)
		{
			luaT_dec<T>(a[i][j]);
		}
		delete [] a[i];
	}
	delete [] a;
}

void LongRange2D::deinit()
{
	if(qXX)
	{
		decAB<dcArray>(qXX, nz); qXX=0;
		decAB<dcArray>(qXY, nz);
		decAB<dcArray>(qXZ, nz);

		decAB<dcArray>(qYX, nz);
		decAB<dcArray>(qYY, nz);
		decAB<dcArray>(qYZ, nz);

		decAB<dcArray>(qZX, nz);
		decAB<dcArray>(qZY, nz);
		decAB<dcArray>(qZZ, nz);


		luaT_dec<dcArray>(hqx);
		luaT_dec<dcArray>(hqy);
		luaT_dec<dcArray>(hqz);

		luaT_dec<dcArray>(hrx);
		luaT_dec<dcArray>(hry);
		luaT_dec<dcArray>(hrz);
	}
	if(XX)
	{
		decAB<dArray>(XX, nz); XX=0;
		decAB<dArray>(XY, nz);
		decAB<dArray>(XZ, nz);

		decAB<dArray>(YX, nz);
		decAB<dArray>(YY, nz);
		decAB<dArray>(YZ, nz);

		decAB<dArray>(ZX, nz);
		decAB<dArray>(ZY, nz);
		decAB<dArray>(ZZ, nz);
	}

	if(g)
	{
		delete [] g;
		g = 0;
	}
}

LongRange2D::~LongRange2D()
{
	deinit();

	if(longrange_ref != LUA_REFNIL)
		luaL_unref(L, LUA_REGISTRYINDEX, longrange_ref);
	if(function_ref != LUA_REFNIL)
		luaL_unref(L, LUA_REGISTRYINDEX, function_ref);
	
	unregisterWS();
}

void LongRange2D::makeNewData()
{
	if(!newDataRequired)
		return;
	newDataRequired = false;

	if(function_ref != LUA_REFNIL)
		lua_rawgeti(L, LUA_REGISTRYINDEX, function_ref);
	else
		luaL_error(L, "make data function not set");

	luaT_push<LongRange2D>(L, this);

	lua_call(L, 1, 0);

	compileRequired = true;
}


void LongRange2D::compile()
{
	if(newDataRequired)
		makeNewData();

	if(!compileRequired)
		return;
	compileRequired = false;

	dcArray* wsZ = new dcArray(nx,ny,1);

	for(int i=0; i<nz; i++)
	{
		for(int j=0; j<nz; j++)
		{
			wsZ->zero();
			arraySetRealPart(wsZ->ddata(), XX[i][j]->ddata(), wsZ->nxyz);  wsZ->fft2DTo(qXX[i][j]);
			arraySetRealPart(wsZ->ddata(), XY[i][j]->ddata(), wsZ->nxyz);  wsZ->fft2DTo(qXY[i][j]);
			arraySetRealPart(wsZ->ddata(), XZ[i][j]->ddata(), wsZ->nxyz);  wsZ->fft2DTo(qXZ[i][j]);

			arraySetRealPart(wsZ->ddata(), YX[i][j]->ddata(), wsZ->nxyz);  wsZ->fft2DTo(qYX[i][j]);
			arraySetRealPart(wsZ->ddata(), YY[i][j]->ddata(), wsZ->nxyz);  wsZ->fft2DTo(qYY[i][j]);
			arraySetRealPart(wsZ->ddata(), YZ[i][j]->ddata(), wsZ->nxyz);  wsZ->fft2DTo(qYZ[i][j]);
			
			arraySetRealPart(wsZ->ddata(), ZX[i][j]->ddata(), wsZ->nxyz);  wsZ->fft2DTo(qZX[i][j]);
			arraySetRealPart(wsZ->ddata(), ZY[i][j]->ddata(), wsZ->nxyz);  wsZ->fft2DTo(qZY[i][j]);
			arraySetRealPart(wsZ->ddata(), ZZ[i][j]->ddata(), wsZ->nxyz);  wsZ->fft2DTo(qZZ[i][j]);

			//prescaling by 1/xy for unscaled fft
			qXX[i][j]->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny)), 0));
			qXY[i][j]->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny)), 0));
			qXZ[i][j]->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny)), 0));
			
			qYX[i][j]->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny)), 0));
			qYY[i][j]->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny)), 0));
			qYZ[i][j]->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny)), 0));
			
			qZX[i][j]->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny)), 0));
			qZY[i][j]->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny)), 0));
			qZZ[i][j]->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny)), 0));

		}
	}
	delete wsZ;
}

bool LongRange2D::apply(SpinSystem* ss)
{
	if(newDataRequired)
		makeNewData();
	
	if(compileRequired)
		compile();

	int slot = markSlotUsed(ss);

	const int nxy = nx*ny;

	doubleComplex one = luaT<doubleComplex>::one();
	
	ss->fft();

	dcArray* sqx = ss->qx;
	dcArray* sqy = ss->qy;
	dcArray* sqz = ss->qz;

	dArray* hx = ss->hx[slot];
	dArray* hy = ss->hy[slot];
	dArray* hz = ss->hz[slot];

	// HX
	ws1->zero();
	for(int d=0; d<nz; d++) //dest
	{
		const int k = d*nxy;
		for(int s=0; s<nz; s++) //src
		{
			const int j = s*nxy;
			arrayScaleMultAdd_o(ws1->ddata(), k, one, qXX[d][s]->ddata(), 0, sqx->ddata(), j, ws1->ddata(), k, nxy); 
			arrayScaleMultAdd_o(ws1->ddata(), k, one, qXY[d][s]->ddata(), 0, sqy->ddata(), j, ws1->ddata(), k, nxy); 
			arrayScaleMultAdd_o(ws1->ddata(), k, one, qXZ[d][s]->ddata(), 0, sqz->ddata(), j, ws1->ddata(), k, nxy); 
		}
	}
	ws1->ifft2DTo(ws2);
	arrayGetRealPart(hx->ddata(),  ws2->ddata(), nxyz);

	// HY
	ws1->zero();
	for(int d=0; d<nz; d++) //dest
	{
		const int k = d*nxy;
		for(int s=0; s<nz; s++) //src
		{
			const int j = s*nxy;
			arrayScaleMultAdd_o(ws1->ddata(), k, one, qYX[d][s]->ddata(), 0, sqx->ddata(), j, ws1->ddata(), k, nxy); 
			arrayScaleMultAdd_o(ws1->ddata(), k, one, qYY[d][s]->ddata(), 0, sqy->ddata(), j, ws1->ddata(), k, nxy); 
			arrayScaleMultAdd_o(ws1->ddata(), k, one, qYZ[d][s]->ddata(), 0, sqz->ddata(), j, ws1->ddata(), k, nxy); 
		}
	}
	ws1->ifft2DTo(ws2);
	arrayGetRealPart(hy->ddata(),  ws2->ddata(), nxyz);

	// HZ
	ws1->zero();
	for(int d=0; d<nz; d++) //dest
	{
		const int k = d*nxy;
		for(int s=0; s<nz; s++) //src
		{
			const int j = s*nxy;
			arrayScaleMultAdd_o(ws1->ddata(), k, one, qZX[d][s]->ddata(), 0, sqx->ddata(), j, ws1->ddata(), k, nxy); 
			arrayScaleMultAdd_o(ws1->ddata(), k, one, qZY[d][s]->ddata(), 0, sqy->ddata(), j, ws1->ddata(), k, nxy); 
			arrayScaleMultAdd_o(ws1->ddata(), k, one, qZZ[d][s]->ddata(), 0, sqz->ddata(), j, ws1->ddata(), k, nxy); 
		}
	}
	ws1->ifft2DTo(ws2);
	arrayGetRealPart(hz->ddata(),  ws2->ddata(), nxyz);

	for(int i=0; i<nz; i++)
	{
		hx->scaleAll_o(g[i] * global_scale, nxy*i, nxy);
		hy->scaleAll_o(g[i] * global_scale, nxy*i, nxy);
		hz->scaleAll_o(g[i] * global_scale, nxy*i, nxy);
	}

	hx->new_device = true;
	hy->new_device = true;
	hz->new_device = true;

	return true;
}





static int l_setstrength(lua_State* L)
{
	LUA_PREAMBLE(LongRange2D, lr, 1);
	int idx = lua_tonumber(L, 2);
	if(idx < 1 || idx > lr->nz)
	{
		return luaL_error(L, "Invalid Layer");
	}
	lr->g[idx-1]= lua_tonumber(L, 3);
	return 0;
}
static int l_getstrength(lua_State* L)
{
	LUA_PREAMBLE(LongRange2D, lr, 1);
	int idx = lua_tonumber(L, 2);
	if(idx < 1 || idx > lr->nz)
	{
		return luaL_error(L, "Invalid Layer");
	}
	
	lua_pushnumber(L, lr->g[idx-1]);

	return 1;
}


static int l_getTensorArray(lua_State* L)
{
	LUA_PREAMBLE(LongRange2D, lr, 1);
	int dest = lua_tonumber(L, 2)-1;
	int src  = lua_tonumber(L, 3)-1;
	const char* AB = lua_tostring(L, 4);
	luaT_push<dArray>(L, lr->getLAB(dest, src, AB));
	return 1;
}
static int l_setTensorArray(lua_State* L)
{
	LUA_PREAMBLE(LongRange2D, lr, 1);
	int dest = lua_tonumber(L, 2)-1;
	int src  = lua_tonumber(L, 3)-1;
	const char* AB = lua_tostring(L, 4);
	dArray* a = luaT_to<dArray>(L, 5);
	lr->setLAB(dest, src, AB, a);
	return 0;
}
static int l_setcompilereqd(lua_State* L) //should move to isCompileRequired and setCompileRequired
{
	LUA_PREAMBLE(LongRange2D, lr, 1);
	if(!lua_isnone(L, 2))
		lr->compileRequired = lua_toboolean(L, 2);
	else
		lr->compileRequired = true;
	return 0;
}
static int l_setnewdatareqd(lua_State* L) //should move to isCompileRequired and setCompileRequired
{
	LUA_PREAMBLE(LongRange2D, lr, 1);
	if(!lua_isnone(L, 2))
		lr->newDataRequired = lua_toboolean(L, 2);
	else
		lr->newDataRequired = true;
	return 0;
}

static int l_setinternaldata(lua_State* L)
{
	LUA_PREAMBLE(LongRange2D, lr, 1);

	if(lua_gettop(L) != 2)
	{
		return luaL_error(L, "require exactly 1 argument");
	}

	luaL_unref(L, LUA_REGISTRYINDEX, lr->longrange_ref);
	lr->longrange_ref = luaL_ref(L, LUA_REGISTRYINDEX);

	return 0;
}

static int l_getinternaldata(lua_State* L)
{
	LUA_PREAMBLE(LongRange2D, lr, 1);

	if(lr->longrange_ref != LUA_REFNIL)
		lua_rawgeti(L, LUA_REGISTRYINDEX, lr->longrange_ref);
	else
		lua_pushnil(L);

	return 1;
}

static int l_setmakefunction(lua_State* L)
{
	LUA_PREAMBLE(LongRange2D, lr, 1);

	if(!lua_isfunction(L, 2))
		return luaL_error(L, "function required as argument");

	luaL_unref(L, LUA_REGISTRYINDEX, lr->function_ref);
	lr->function_ref = luaL_ref(L, LUA_REGISTRYINDEX);
	return 0;
}
static int l_getmakefunction(lua_State* L)
{
	LUA_PREAMBLE(LongRange2D, lr, 1);

	if(!lua_isfunction(L, 2))
		return luaL_error(L, "function required as argument");

	if(lr->function_ref != LUA_REFNIL)
		lua_rawgeti(L, LUA_REGISTRYINDEX, lr->function_ref);
	else
		lua_pushnil(L);
	return 0;
}
static int l_makedata(lua_State* L)
{
	LUA_PREAMBLE(LongRange2D, lr, 1);
	lr->makeNewData();
	return 0;
}

int LongRange2D::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates a Long Range field for a *SpinSystem*. This is a base class used by other operators. This operator by itself does nothing.");
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if(func == l_makedata)
	{
		lua_pushstring(L, "Generates new internal data if required");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_setstrength)
	{
		lua_pushstring(L, "Set the strength of the Long Range Field at a given layer");
		lua_pushstring(L, "1 Integer, 1 number: strength of the field at layer");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_getstrength)
	{
		lua_pushstring(L, "Get the strength of the Long Range Field at a given layer");
		lua_pushstring(L, "1 Integer: Layer");
		lua_pushstring(L, "1 Number: strength of the field");
		return 3;
	}

	if(func == l_getTensorArray)
	{
		lua_pushstring(L, "Get the tensor array for one layer interacting with another. Layers indices are base 1.");
		lua_pushstring(L, "1 Integer, 1 Integer, 1 String: Destination layer, Source Layer, Tensor name.");
		lua_pushstring(L, "1 Array: Tensor");
		return 3;
	}
	if(func == l_setTensorArray)
	{
		lua_pushstring(L, "Set the tensor array for one layer interacting with another. Layers indices are base 1.");
		lua_pushstring(L, "1 Integer, 1 Integer, 1 String, 1 Array: Destination layer, Source Layer, Tensor name and array.");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_setcompilereqd)
	{
		lua_pushstring(L, "Set internal compile required state. Must be set to true if any of the interaction tensors were modified.");
		lua_pushstring(L, "1 Optional Boolean (default true): new internal flag state");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_setnewdatareqd)
	{
		lua_pushstring(L, "Set internal new data required state. This is used in some internal routines where the data is set manually rather than calculated");
		lua_pushstring(L, "1 Optional Boolean (default true): new internal data required flag state");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_setinternaldata)
	{
		lua_pushstring(L, "Set internal data used to build tensors");
		lua_pushstring(L, "1 value: usually a table holding data");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_getinternaldata)
	{
		lua_pushstring(L, "Get internal data used to build tensors");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 value: usually a table holding data");
		return 3;
	}
	if(func == l_setmakefunction)
	{
		lua_pushstring(L, "Set internal function used to build tensors");
		lua_pushstring(L, "1 function: argument of function should be a LongRange2D object");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_getmakefunction)
	{
		lua_pushstring(L, "Get internal function used to build tensors");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 function: argument of function should be a LongRange2D object");
		return 3;
	}
	return SpinOperation::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* LongRange2D::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, SpinOperation::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"setStrength",   l_setstrength},
		{"strength",      l_getstrength},
		{"tensorArray",   l_getTensorArray},
		{"setTensorArray",l_setTensorArray},
		{"setCompileRequired",       l_setcompilereqd},
		{"setNewDataRequired",       l_setnewdatareqd},
		{"setInternalData", l_setinternaldata},
		{"internalData", l_getinternaldata},
		{"setMakeDataFunction", l_setmakefunction},
		{"makeDataFunction", l_getmakefunction},
		{"makeData", l_makedata},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}





extern "C"
{
LONGRANGE2D_API int lib_register(lua_State* L);
LONGRANGE2D_API int lib_version(lua_State* L);
LONGRANGE2D_API const char* lib_name(lua_State* L);
LONGRANGE2D_API int lib_main(lua_State* L);
}

#include "longrange2d_luafuncs.h"

static int l_getmetatable(lua_State* L)
{
	if(!lua_isstring(L, 1))
		return luaL_error(L, "First argument must be a metatable name");
	luaL_getmetatable(L, lua_tostring(L, 1));
	return 1;
}

LONGRANGE2D_API int lib_register(lua_State* L)
{
	luaT_register<LongRange2D>(L);

	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");
	if(luaL_dostring(L, __longrange2d_luafuncs()))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return luaL_error(L, lua_tostring(L, -1));
	}

	lua_pushnil(L);
	lua_setglobal(L, "maglua_getmetatable");

	return 0;
}


LONGRANGE2D_API int lib_version(lua_State* L)
{
	return __revi;
}

LONGRANGE2D_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "LongRange2D-Cuda";
#else
	return "LongRange2D-Cuda-Debug";
#endif
}

LONGRANGE2D_API int lib_main(lua_State* L)
{
	return 0;
}



