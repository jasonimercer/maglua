/******************************************************************************
* Copyright (C) 2008-2013 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#include "spinoperationappliedfield_heterogeneous.h"
#include "spinsystem.h"

#include <stdlib.h>

AppliedField_Heterogeneous::AppliedField_Heterogeneous(int nx, int ny, int nz)
	: SpinOperation(nx, ny, nz, hash32(AppliedField_Heterogeneous::typeName()))
{
	hx = 0;
	hy = 0;
	hz = 0;
}

int AppliedField_Heterogeneous::luaInit(lua_State* L)
{
	int r = SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	
	hx = luaT_inc<dArray>(new dArray(nx,ny,nz));
	hy = luaT_inc<dArray>(new dArray(nx,ny,nz));
	hz = luaT_inc<dArray>(new dArray(nx,ny,nz));
	
	return r;
}

	
void AppliedField_Heterogeneous::encode(buffer* b)
{
	SpinOperation::encode(b);
	char version = 0;
	encodeChar(version, b);
	hx->encode(b);
	hy->encode(b);
	hz->encode(b);
}

int  AppliedField_Heterogeneous::decode(buffer* b)
{
	SpinOperation::decode(b);
	char version = decodeChar(b);
	if(version == 0)
	{
		luaT_dec<dArray>(hx);
		luaT_dec<dArray>(hy);
		luaT_dec<dArray>(hz);
		
		hx = luaT_inc<dArray>(new dArray(nx,ny,nz));
		hy = luaT_inc<dArray>(new dArray(nx,ny,nz));
		hz = luaT_inc<dArray>(new dArray(nx,ny,nz));
	
		hx->decode(b);
		hy->decode(b);
		hz->decode(b);
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}
	return 0;
}

AppliedField_Heterogeneous::~AppliedField_Heterogeneous()
{
	luaT_dec<dArray>(hx);
	luaT_dec<dArray>(hy);
	luaT_dec<dArray>(hz);
}

bool AppliedField_Heterogeneous::apply(SpinSystem** ss, int n)
{
	for(int i=0; i<n; i++)
		apply(ss[i]);
	return true;
}

bool AppliedField_Heterogeneous::apply(SpinSystem* ss)
{
	int slot = markSlotUsed(ss);

	dArray::pairwiseScaleAdd(ss->hx[slot], 1, ss->hx[slot], global_scale, hx);
	dArray::pairwiseScaleAdd(ss->hy[slot], 1, ss->hy[slot], global_scale, hy);
	dArray::pairwiseScaleAdd(ss->hz[slot], 1, ss->hz[slot], global_scale, hz);

	return true;
}

static int l_sx(lua_State* L)
{
	LUA_PREAMBLE(AppliedField_Heterogeneous, af, 1);
	LUA_PREAMBLE(dArray, a, 2);
	luaT_inc<dArray>(a);
	luaT_dec<dArray>(af->hx);
	af->hx = a;
	return 0;
}
static int l_sy(lua_State* L)
{
	LUA_PREAMBLE(AppliedField_Heterogeneous, af, 1);
	LUA_PREAMBLE(dArray, a, 2);
	luaT_inc<dArray>(a);
	luaT_dec<dArray>(af->hy);
	af->hy = a;
	return 0;
}
static int l_sz(lua_State* L)
{
	LUA_PREAMBLE(AppliedField_Heterogeneous, af, 1);
	LUA_PREAMBLE(dArray, a, 2);
	luaT_inc<dArray>(a);
	luaT_dec<dArray>(af->hz);
	af->hz = a;
	return 0;
}


static int l_gx(lua_State* L)
{
	LUA_PREAMBLE(AppliedField_Heterogeneous, af, 1);
	luaT_push<dArray>(L, af->hx);
	return 1;
}
static int l_gy(lua_State* L)
{
	LUA_PREAMBLE(AppliedField_Heterogeneous, af, 1);
	luaT_push<dArray>(L, af->hy);
	return 1;
}
static int l_gz(lua_State* L)
{
	LUA_PREAMBLE(AppliedField_Heterogeneous, af, 1);
	luaT_push<dArray>(L, af->hz);
	return 1;
}


int AppliedField_Heterogeneous::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Applies an external field to a *SpinSystem* with different values at each location.");
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);	

	if(func == l_sx)
	{
		lua_pushstring(L, "Set the X fields to an external array.");
		lua_pushstring(L, "1 Array of Double Precision Numbers: New field array.");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_sy)
	{
		lua_pushstring(L, "Set the Y fields to an external array.");
		lua_pushstring(L, "1 Array of Double Precision Numbers: New field array.");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_sz)
	{
		lua_pushstring(L, "Set the Z fields to an external array.");
		lua_pushstring(L, "1 Array of Double Precision Numbers: New field array.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_gx)
	{
		lua_pushstring(L, "Get the internal X field array.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Array of Double Precision Numbers: Field array.");
		return 3;
	}
	if(func == l_gy)
	{
		lua_pushstring(L, "Get the internal Y field array.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Array of Double Precision Numbers: Field array.");
		return 3;
	}
	if(func == l_gz)
	{
		lua_pushstring(L, "Get the internal Z field array.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Array of Double Precision Numbers: Field array.");
		return 3;
	}


	return SpinOperation::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* AppliedField_Heterogeneous::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, SpinOperation::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"fieldArrayX",            l_gx},
		{"fieldArrayY",            l_gy},
		{"fieldArrayZ",            l_gz},
		{"setFieldArrayX",         l_sx},
		{"setFieldArrayY",         l_sy},
		{"setFieldArrayZ",         l_sz},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}





