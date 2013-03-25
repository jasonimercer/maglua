/******************************************************************************
* Copyright (C) 2012 Jason Mercer.  All rights reserved.
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
#include "spinoperationlongrange3d.h"
#include "luamigrate.h"

#include <stdlib.h>
#include <math.h>

#if defined NDEBUG || defined __OPTIMIZE__
#define DDD
#else
#define DDD printf("(%s:%i)\n", __FILE__, __LINE__);
#endif


// LongRange3D::LongRange3D(int nx, int ny, int nz)
// 	: SpinOperation("LongRange3D", DIPOLE_SLOT, nx, ny, nz, hash32("LongRange3D"))
	
LongRange3D::LongRange3D(const char* Name, const int field_slot, int nx, int ny, int nz, const int encode_tag)
	: SpinOperation(Name, field_slot, nx, ny, nz, encode_tag)
{
	registerWS();
    qXX = 0;
    XX = 0;
	ws1 = 0;
	g = 0;

	longrange_ref = LUA_REFNIL;
	function_ref = LUA_REFNIL;

	XX = 0;
	XY = 0;
	XZ = 0;

	YX = 0;
	YY = 0;
	YZ = 0;

	ZX = 0;
	ZY = 0;
	ZZ = 0;

	compileRequired = true;
	newDataRequired = true;
	
}

LongRange3D::~LongRange3D()
{
	deinit();

	if(longrange_ref != LUA_REFNIL)
		luaL_unref(L, LUA_REGISTRYINDEX, longrange_ref);
	longrange_ref = LUA_REFNIL;
	if(function_ref != LUA_REFNIL)
		luaL_unref(L, LUA_REGISTRYINDEX, function_ref);
	function_ref = LUA_REFNIL;
	
	unregisterWS();
}

int LongRange3D::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	LongRange3D::init();
	return 0;	
}

void LongRange3D::encode(buffer* b)
{
	SpinOperation::encode(b);
	char version = 0;
	encodeChar(version, b);
	
	encodeDouble(g, b);
	
	int ref[2];
	ref[0] = longrange_ref;
	ref[1] = function_ref;

	for(int i=0; i<2; i++)
	{
		if(ref[i] != LUA_REFNIL)
		{
			lua_rawgeti(L, LUA_REGISTRYINDEX, ref[i]);
		}
		else
		{
			lua_pushnil(L);
		}

		_exportLuaVariable(L, lua_gettop(L), b);
		lua_pop(L, 1);
	}
}

int  LongRange3D::decode(buffer* b)
{
	SpinOperation::decode(b);
	
	char version = decodeChar(b);
	if(version == 0)
	{
		g = decodeDouble(b);
		
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

		
		deinit();
		init();
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}
	return 0;
}

dArray* LongRange3D::getAB(const char* AB)
{
	if(strncasecmp(AB, "xx", 2) == 0)
		return XX;
	if(strncasecmp(AB, "xy", 2) == 0)
		return XY;
	if(strncasecmp(AB, "xz", 2) == 0)
		return XZ;

	if(strncasecmp(AB, "yx", 2) == 0)
		return YX;
	if(strncasecmp(AB, "yy", 2) == 0)
		return YY;
	if(strncasecmp(AB, "yz", 2) == 0)
		return YZ;
	
	if(strncasecmp(AB, "zx", 2) == 0)
		return ZX;
	if(strncasecmp(AB, "zy", 2) == 0)
		return ZY;
	if(strncasecmp(AB, "zz", 2) == 0)
		return ZZ;

	return 0;
}

void LongRange3D::setAB(const char* AB, dArray* newArray)
{
	dArray** a[9];
	a[0] = &XX;	a[1] = &XY;	a[2] = &XZ;
	a[3] = &YX;	a[4] = &YY;	a[5] = &YZ;
	a[6] = &ZX;	a[7] = &ZY;	a[8] = &ZZ;
	
	const char* n[9] = {
		"XX", "XY", "XZ",
		"YX", "YY", "YZ",
		"ZX", "ZY", "ZZ"};
		
	luaT_inc<dArray>(newArray);
	for(int i=0; i<9; i++)
	{
		if(strncasecmp(AB, n[i], 2) == 0)
		{
			luaT_dec<dArray>(*a[i]);
			*a[i] = newArray;
		}
	}
}


void LongRange3D::init()
{
	if(XX) return;

	deinit();
	nxyz = nx*ny*nz;
	hqx = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	hqy = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	hqz = luaT_inc<dcArray>(new dcArray(nx,ny,nz));

	hrx = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	hry = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	hrz = luaT_inc<dcArray>(new dcArray(nx,ny,nz));

	qXX = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qXY = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qXZ = luaT_inc<dcArray>(new dcArray(nx,ny,nz));

	qYX = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qYY = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qYZ = luaT_inc<dcArray>(new dcArray(nx,ny,nz));

	qZX = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qZY = luaT_inc<dcArray>(new dcArray(nx,ny,nz));
	qZZ = luaT_inc<dcArray>(new dcArray(nx,ny,nz));

	XX = luaT_inc<dArray>(new dArray(nx,ny,nz));
	XY = luaT_inc<dArray>(new dArray(nx,ny,nz));
	XZ = luaT_inc<dArray>(new dArray(nx,ny,nz));
	YX = luaT_inc<dArray>(new dArray(nx,ny,nz));
	YY = luaT_inc<dArray>(new dArray(nx,ny,nz));
	YZ = luaT_inc<dArray>(new dArray(nx,ny,nz));
	ZX = luaT_inc<dArray>(new dArray(nx,ny,nz));
	ZY = luaT_inc<dArray>(new dArray(nx,ny,nz));
	ZZ = luaT_inc<dArray>(new dArray(nx,ny,nz));

	XX->zero();
	XY->zero();
	XZ->zero();
	YX->zero();
	YY->zero();
	YZ->zero();
	ZX->zero();
	ZY->zero();
	ZZ->zero();
	
	ws1 = getWSdcArray(nx,ny,nz, hash32("SpinOperation::apply_1"));
	ws2 = getWSdcArray(nx,ny,nz, hash32("SpinOperation::apply_2"));
	wsX = getWSdcArray(nx,ny,nz, hash32("SpinOperation::apply_3"));
	wsY = getWSdcArray(nx,ny,nz, hash32("SpinOperation::apply_4"));
	wsZ = getWSdcArray(nx,ny,nz, hash32("SpinOperation::apply_5"));

}

static int offsetOK(int nx, int ny, int nz,  int x, int y, int z, int& offset)
{
	if(x<0  || x >= nx) return 0;
	if(y<0  || y >= ny) return 0;
	if(z<0  || z >= nz) return 0;
	
	offset = x + y*nx + z*nx*ny;
	return 1;
}


void LongRange3D::deinit()
{
	if(qXX)
	{
		luaT_dec<dcArray>(qXX); qXX=0;
		luaT_dec<dcArray>(qXY);
		luaT_dec<dcArray>(qXZ);

		luaT_dec<dcArray>(qYX);
		luaT_dec<dcArray>(qYY);
		luaT_dec<dcArray>(qYZ);

		luaT_dec<dcArray>(qZX);
		luaT_dec<dcArray>(qZY);
		luaT_dec<dcArray>(qZZ);

		luaT_dec<dcArray>(hqx);
		luaT_dec<dcArray>(hqy);
		luaT_dec<dcArray>(hqz);

		luaT_dec<dcArray>(hrx);
		luaT_dec<dcArray>(hry);
		luaT_dec<dcArray>(hrz);
	}
	if(XX)
	{
		luaT_dec<dArray>(XX); XX=0;
		luaT_dec<dArray>(XY); XY=0;
		luaT_dec<dArray>(XZ); XZ=0;
		luaT_dec<dArray>(YX); YX=0;
		luaT_dec<dArray>(YY); YY=0;
		luaT_dec<dArray>(YZ); YZ=0;
		luaT_dec<dArray>(ZX); ZX=0;
		luaT_dec<dArray>(ZY); ZY=0;
		luaT_dec<dArray>(ZZ); ZZ=0;
	}
	
	
	if(ws1)
	{
		ws1 = 0;
		ws2 = 0;
		wsX = 0;
		wsY = 0;
		wsZ = 0;
	}
	
	
}
void LongRange3D::makeNewData()
{
	if(!newDataRequired)
		return;
	newDataRequired = false;

	if(function_ref != LUA_REFNIL)
		lua_rawgeti(L, LUA_REGISTRYINDEX, function_ref);
	else
		luaL_error(L, "make data function not set");

	luaT_push<LongRange3D>(L, this);

	lua_call(L, 1, 0);

	compileRequired = true;
}


void LongRange3D::compile()
{
	if(newDataRequired)
		makeNewData();

	if(!compileRequired)
		return;
	compileRequired = false;

	dcArray* wsZ = ws1;

	arraySetRealPart(wsZ->ddata(), XX->ddata(), wsZ->nxyz);  wsZ->fft3DTo(qXX);
	arraySetRealPart(wsZ->ddata(), XY->ddata(), wsZ->nxyz);  wsZ->fft3DTo(qXY);
	arraySetRealPart(wsZ->ddata(), XZ->ddata(), wsZ->nxyz);  wsZ->fft3DTo(qXZ);

	arraySetRealPart(wsZ->ddata(), YX->ddata(), wsZ->nxyz);  wsZ->fft3DTo(qYX);
	arraySetRealPart(wsZ->ddata(), YY->ddata(), wsZ->nxyz);  wsZ->fft3DTo(qYY);
	arraySetRealPart(wsZ->ddata(), YZ->ddata(), wsZ->nxyz);  wsZ->fft3DTo(qYZ);
			
	arraySetRealPart(wsZ->ddata(), ZX->ddata(), wsZ->nxyz);  wsZ->fft3DTo(qZX);
	arraySetRealPart(wsZ->ddata(), ZY->ddata(), wsZ->nxyz);  wsZ->fft3DTo(qZY);
	arraySetRealPart(wsZ->ddata(), ZZ->ddata(), wsZ->nxyz);  wsZ->fft3DTo(qZZ);

	//prescaling by 1/xyz for unscaled fft
	qXX->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny*nz)), 0));
	qXY->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny*nz)), 0));
	qXZ->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny*nz)), 0));
	
	qYX->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny*nz)), 0));
	qYY->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny*nz)), 0));
	qYZ->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny*nz)), 0));
	
	qZX->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny*nz)), 0));
	qZY->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny*nz)), 0));
	qZZ->scaleAll(make_cuDoubleComplex(1.0/((double)(nx*ny*nz)), 0));

}

bool LongRange3D::apply(SpinSystem* ss)
{
	if(newDataRequired)
		makeNewData();
	
	if(compileRequired)
		compile();

	markSlotUsed(ss);

	const int nxyz = nx*ny*nz;

	doubleComplex one = luaT<doubleComplex>::one();

	arraySetRealPart(ws1->ddata(), ss->x->ddata(), ws1->nxyz); ws1->fft3DTo(wsX);
	arraySetRealPart(ws1->ddata(), ss->y->ddata(), ws1->nxyz); ws1->fft3DTo(wsY);
	arraySetRealPart(ws1->ddata(), ss->z->ddata(), ws1->nxyz); ws1->fft3DTo(wsZ);
	
	dcArray* sqx = wsX;
	dcArray* sqy = wsY;
	dcArray* sqz = wsZ;

	dArray* hx = ss->hx[slot];
	dArray* hy = ss->hy[slot];
	dArray* hz = ss->hz[slot];
	
	// HX
	ws1->zero();
// 	printf("(%s:%i)   %p %p %p %p\n", __FILE__, __LINE__, ws1->ddata(), qXX->ddata(), sqx->ddata(), ws1->ddata());
	arrayScaleMultAdd_o(ws1->ddata(), 0, one, qXX->ddata(), 0, sqx->ddata(), 0, ws1->ddata(), 0, nxyz); 
	arrayScaleMultAdd_o(ws1->ddata(), 0, one, qXY->ddata(), 0, sqy->ddata(), 0, ws1->ddata(), 0, nxyz); 
	arrayScaleMultAdd_o(ws1->ddata(), 0, one, qXZ->ddata(), 0, sqz->ddata(), 0, ws1->ddata(), 0, nxyz); 
 	ws1->ifft3DTo(ws2);
 	arrayGetRealPart(hx->ddata(),  ws2->ddata(), nxyz);

	// HY
	ws1->zero();
	arrayScaleMultAdd_o(ws1->ddata(), 0, one, qYX->ddata(), 0, sqx->ddata(), 0, ws1->ddata(), 0, nxyz); 
	arrayScaleMultAdd_o(ws1->ddata(), 0, one, qYY->ddata(), 0, sqy->ddata(), 0, ws1->ddata(), 0, nxyz); 
	arrayScaleMultAdd_o(ws1->ddata(), 0, one, qYZ->ddata(), 0, sqz->ddata(), 0, ws1->ddata(), 0, nxyz); 
	ws1->ifft3DTo(ws2);
	arrayGetRealPart(hy->ddata(),  ws2->ddata(), nxyz);
	
	// HZ
	ws1->zero();
	arrayScaleMultAdd_o(ws1->ddata(), 0, one, qZX->ddata(), 0, sqx->ddata(), 0, ws1->ddata(), 0, nxyz); 
	arrayScaleMultAdd_o(ws1->ddata(), 0, one, qZY->ddata(), 0, sqy->ddata(), 0, ws1->ddata(), 0, nxyz); 
	arrayScaleMultAdd_o(ws1->ddata(), 0, one, qZZ->ddata(), 0, sqz->ddata(), 0, ws1->ddata(), 0, nxyz); 
	ws1->ifft3DTo(ws2);
	arrayGetRealPart(hz->ddata(),  ws2->ddata(), nxyz);

	hx->scaleAll(g * global_scale);
	hy->scaleAll(g * global_scale);
	hz->scaleAll(g * global_scale);

	return true;
}











static int l_setstrength(lua_State* L)
{
	LUA_PREAMBLE(LongRange3D, lr, 1);
	lr->g = lua_tonumber(L, 2);
	return 0;
}
static int l_getstrength(lua_State* L)
{
	LUA_PREAMBLE(LongRange3D, lr, 1);
	lua_pushnumber(L, lr->g);

	return 1;
}


static int l_getTensorArray(lua_State* L)
{
	LUA_PREAMBLE(LongRange3D, lr, 1);
	const char* AB = lua_tostring(L, 2);
	luaT_push<dArray>(L, lr->getAB(AB));
	return 1;
}
static int l_setTensorArray(lua_State* L)
{
	LUA_PREAMBLE(LongRange3D, lr, 1);
	const char* AB = lua_tostring(L, 2);
	dArray* a = luaT_to<dArray>(L, 3);
	lr->setAB(AB, a);
	return 0;
}
static int l_setcompilereqd(lua_State* L) //should move to isCompileRequired and setCompileRequired
{
	LUA_PREAMBLE(LongRange3D, lr, 1);
	if(!lua_isnone(L, 2))
		lr->compileRequired = lua_toboolean(L, 2);
	else
		lr->compileRequired = true;
	return 0;
}
static int l_setrewdatareqd(lua_State* L) //should move to isCompileRequired and setCompileRequired
{
	LUA_PREAMBLE(LongRange3D, lr, 1);
	if(!lua_isnone(L, 2))
		lr->newDataRequired = lua_toboolean(L, 2);
	else
		lr->newDataRequired = true;
	return 0;
}

static int l_setinternaldata(lua_State* L)
{
	LUA_PREAMBLE(LongRange3D, lr, 1);

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
	LUA_PREAMBLE(LongRange3D, lr, 1);

	if(lr->longrange_ref != LUA_REFNIL)
		lua_rawgeti(L, LUA_REGISTRYINDEX, lr->longrange_ref);
	else
		lua_pushnil(L);

	return 1;
}

static int l_setmakefunction(lua_State* L)
{
	LUA_PREAMBLE(LongRange3D, lr, 1);

	if(!lua_isfunction(L, 2))
		return luaL_error(L, "function required as argument");

	luaL_unref(L, LUA_REGISTRYINDEX, lr->function_ref);
	lr->function_ref = luaL_ref(L, LUA_REGISTRYINDEX);
	return 0;
}
static int l_getmakefunction(lua_State* L)
{
	LUA_PREAMBLE(LongRange3D, lr, 1);

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
	LUA_PREAMBLE(LongRange3D, lr, 1);
	lr->makeNewData();
	return 0;
}

int LongRange3D::help(lua_State* L)
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
		lua_pushstring(L, "Set the strength of the Long Range Field ");
		lua_pushstring(L, "1 Number: Strength of the field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_getstrength)
	{
		lua_pushstring(L, "Get the strength of the Long Range Field at a given layer");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: strength of the field");
		return 3;
	}

	if(func == l_getTensorArray)
	{
		lua_pushstring(L, "Get the tensor array.");
		lua_pushstring(L, "1 String: Tensor name (\"XX\", \"XY\", ... \"ZZ\").");
		lua_pushstring(L, "1 Array: Tensor");
		return 3;
	}
	if(func == l_setTensorArray)
	{
		lua_pushstring(L, "Set the tensor array.");
		lua_pushstring(L, "1 String, 1 Array: Tensor name and array.");
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
	if(func == l_setrewdatareqd)
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
		lua_pushstring(L, "1 function: argument of function should be a LongRange3D object");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_getmakefunction)
	{
		lua_pushstring(L, "Get internal function used to build tensors");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 function: argument of function should be a LongRange3D object");
		return 3;
	}
	return SpinOperation::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* LongRange3D::luaMethods()
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
		{"setNewDataRequired",       l_setrewdatareqd},
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




#include "longrange3d_luafuncs.h"
static int l_getmetatable(lua_State* L)
{
    if(!lua_isstring(L, 1))
        return luaL_error(L, "First argument must be a metatable name");
    luaL_getmetatable(L, lua_tostring(L, 1));
    return 1;
}

extern "C"
{
LONGRANGE_API int lib_register(lua_State* L);
LONGRANGE_API int lib_version(lua_State* L);
LONGRANGE_API const char* lib_name(lua_State* L);
LONGRANGE_API int lib_main(lua_State* L);
}

LONGRANGE_API int lib_register(lua_State* L)
{
	luaT_register<LongRange3D>(L);
	
	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");
	if(luaL_dostring(L, __longrange3d_luafuncs()))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return luaL_error(L, lua_tostring(L, -1));
	}

	lua_pushnil(L);
	lua_setglobal(L, "maglua_getmetatable");
	
	return 0;
}

LONGRANGE_API int lib_version(lua_State* L)
{
	return __revi;
}

LONGRANGE_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "LongRange3D-Cuda";
#else
	return "LongRange3D-Cuda-Debug";
#endif
}

LONGRANGE_API int lib_main(lua_State* L)
{
	return 0;
}



