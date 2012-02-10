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

#include "spinoperationdipole.h"
#include "spinsystem.h"
#include "dipolesupport.h"
#include "info.h"
#ifndef WIN32
#include <strings.h>
#endif

#include <stdlib.h>
#include <math.h>

Dipole::Dipole(int nx, int ny, int nz)
	: LongRange("Dipole", DIPOLE_SLOT, nx, ny, nz, ENCODE_DIPOLE)
{

}




void Dipole::encode(buffer* b)
{
	encodeInteger(nx, b);
	encodeInteger(ny, b);
	encodeInteger(nz, b);
	encodeInteger(gmax, b);
	encodeDouble(g, b);

	for(int i=0; i<9; i++)
	{
		encodeDouble(ABC[i], b);
	}
}

int  Dipole::decode(buffer* b)
{
	deinit();
	
	nx = decodeInteger(b);
	ny = decodeInteger(b);
	nz = decodeInteger(b);
	gmax = decodeInteger(b);
	g = decodeDouble(b);
	nxyz = nx*ny*nz;

	for(int i=0; i<9; i++)
	{
		ABC[i] = decodeDouble(b);
	}

	return 0;
}

Dipole::~Dipole()
{
}

void Dipole::loadMatrixFunction(double* XX, double* XY, double* XZ, double* YY, double* YZ, double* ZZ)
{
	dipoleLoad(
		nx, ny, nz,
		gmax, ABC,
		XX, XY, XZ,
		YY, YZ, ZZ);
}







Dipole* checkDipole(lua_State* L, int idx)
{
	Dipole** pp = (Dipole**)luaL_checkudata(L, idx, "MERCER.dipole");
    luaL_argcheck(L, pp != NULL, 1, "`Dipole' expected");
    return *pp;
}

void lua_pushDipole(lua_State* L, Encodable* _dip)
{
	Dipole* dip = dynamic_cast<Dipole*>(_dip);
	if(!dip) return;
	dip->refcount++;
	Dipole** pp = (Dipole**)lua_newuserdata(L, sizeof(Dipole**));
	
	*pp = dip;
	luaL_getmetatable(L, "MERCER.dipole");
	lua_setmetatable(L, -2);
}

int l_dip_new(lua_State* L)
{
	int n[3];
	lua_getnewargs(L, n, 1);

	lua_pushDipole(L, new Dipole(n[0], n[1], n[2]));
	return 1;
}


int l_dip_setstrength(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;

	dip->g = lua_tonumber(L, 2);
	return 0;
}

int l_dip_gc(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;

	dip->refcount--;
	if(dip->refcount == 0)
		delete dip;
	
	return 0;
}

int l_dip_apply(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;
	SpinSystem* ss = checkSpinSystem(L, 2);
	
	if(!dip->apply(ss))
		return luaL_error(L, dip->errormsg.c_str());
	
	return 0;
}

// int l_dip_threadapply(lua_State* L)
// {
// 	Dipole* dip = checkDipole(L, 1);
// 	if(!dip) return 0;
// 	SpinSystem* ss = checkSpinSystem(L, 2);
// 	
// 	dip->threadApply(ss);
// 	
// 	return 0;
// }

int l_dip_getstrength(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;

	lua_pushnumber(L, dip->g);

	return 1;
}
int l_dip_setunitcell(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;

	double A[3];
	double B[3];
	double C[3];
	
	int r1 = lua_getNdouble(L, 3, A, 2, 0);
	int r2 = lua_getNdouble(L, 3, B, 2+r1, 0);
	int r3 = lua_getNdouble(L, 3, C, 2+r1+r2, 0);
	
	for(int i=0; i<3; i++)
	{
		dip->ABC[i+0] = A[i];
		dip->ABC[i+3] = B[i];
		dip->ABC[i+6] = C[i];
	}

	return 0;
}
int l_dip_getunitcell(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;

	double* ABC[3];
	ABC[0] = &(dip->ABC[0]);
	ABC[1] = &(dip->ABC[3]);
	ABC[2] = &(dip->ABC[6]);
	
	for(int i=0; i<3; i++)
	{
		lua_newtable(L);
		for(int j=0; j<3; j++)
		{
			lua_pushinteger(L, j+1);
			lua_pushnumber(L, ABC[i][j]);
			lua_settable(L, -3);
		}
	}
	
	return 3;
}
static int l_dip_settrunc(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;

	lua_getglobal(L, "math");
	lua_pushstring(L, "huge");
	lua_gettable(L, -2);
	lua_pushvalue(L, 2);
	int huge = lua_equal(L, -2, -1);
	
	if(huge)
	{
		dip->gmax = -1;
	}
	else
	{
		dip->gmax = lua_tointeger(L, 2);
	}
	return 0;
}
static int l_dip_gettrunc(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;

	if(dip->gmax == -1)
	{
		lua_getglobal(L, "math");
		lua_pushstring(L, "huge");
		lua_gettable(L, -2);
		lua_remove(L, -2);//remove table (not really needed);
	}
	else
		lua_pushnumber(L, dip->gmax);

	return 1;
}

static int l_setmatrix(lua_State* L)
{
	Dipole* p = checkDipole(L, 1);
	if(!p) return 0;
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
	p->setAB(mat, offset[0], offset[1], offset[2], val);

	return 0;
}

static int l_getmatrix(lua_State* L)
{
	Dipole* p = checkDipole(L, 1);
	if(!p) return 0;
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
	double val = p->getAB(mat, offset[0], offset[1], offset[2]);
	lua_pushnumber(L, val);
	return 1;
}

static int l_dip_tostring(lua_State* L)
{
	Dipole* dip = checkDipole(L, 1);
	if(!dip) return 0;
	
	lua_pushfstring(L, "Dipole (%dx%dx%d)", dip->nx, dip->ny, dip->nz);
	
	return 1;
}

static int l_dip_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.dipole");
	return 1;
}

static int l_dip_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates the dipolar field of a *SpinSystem*");
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
		return luaL_error(L, "help expects zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if(func == l_dip_new)
	{
		lua_pushstring(L, "Create a new Dipole Operator.");
		lua_pushstring(L, "1 *3Vector*: system size"); 
		lua_pushstring(L, "1 Dipole object");
		return 3;
	}
	
	
	if(func == l_dip_apply)
	{
		lua_pushstring(L, "Calculate the dipolar field of a *SpinSystem*");
		lua_pushstring(L, "1 *SpinSystem*: This spin system will receive the field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_dip_setstrength)
	{
		lua_pushstring(L, "Set the strength of the Dipolar Field");
		lua_pushstring(L, "1 number: strength of the field");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_dip_getstrength)
	{
		lua_pushstring(L, "Get the strength of the Dipolar Field");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 number: strength of the field");
		return 3;
	}
	
	if(func == l_dip_setunitcell)
	{
		lua_pushstring(L, "Set the unit cell of a lattice site");
		lua_pushstring(L, "3 *3Vector*: The A, B and C vectors defining the unit cell. By default, this is {1,0,0},{0,1,0},{0,0,1} or a cubic system.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_dip_getunitcell)
	{
		lua_pushstring(L, "Get the unit cell of a lattice site");
		lua_pushstring(L, "");
		lua_pushstring(L, "3 tables: The A, B and C vectors defining the unit cell. By default, this is {1,0,0},{0,1,0},{0,0,1} or a cubic system.");
		return 3;
	}

	if(func == l_dip_settrunc)
	{
		lua_pushstring(L, "Set the truncation distance in spins of the dipolar sum.");
		lua_pushstring(L, "1 Integers: Radius of spins to sum out to. If set to math.huge then extrapolation will be used to approximate infinite radius.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_dip_gettrunc)
	{
		lua_pushstring(L, "Get the truncation distance in spins of the dipolar sum.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integers: Radius of spins to sum out to.");
		return 3;
	}
	if(func == l_getmatrix)
	{
		lua_pushstring(L, "Get an element of an interaction matrix");
		lua_pushstring(L, "1 string, 1 *3Vector*: The string indicates which AB matrix to access. Can be XX, XY, XZ, YY, YZ or ZZ. The *3Vector* indexes into the matrix. Note: indexes are zero-based and are interpreted as offsets.");
		lua_pushstring(L, "1 number: The fetched value.");
		return 3;
	}

	if(func == l_setmatrix)
	{
		lua_pushstring(L, "Set an element of an interaction matrix");
		lua_pushstring(L, "1 string, 1 *3Vector*, 1 number: The string indicates which AB matrix to access. Can be XX, XY, XZ, YY, YZ or ZZ. The *3Vector* indexes into the matrix. The number is the value that is set at the index. Note: indexes are zero-based and are interpreted as offsets.");
		lua_pushstring(L, "");
		return 3;
	}
	
	return 0;
}

static Encodable* newThing()
{
	return new Dipole;
}

void registerDipole(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_dip_gc},
		{"__tostring",   l_dip_tostring},
		{"apply",        l_dip_apply},
// 		{"threadApply",  l_dip_threadapply},
		{"setStrength",  l_dip_setstrength},
		{"strength",     l_dip_getstrength},
		{"setUnitCell",  l_dip_setunitcell},
		{"unitCell",     l_dip_getunitcell},
		{"setTruncation",l_dip_settrunc},
		{"truncation",   l_dip_gettrunc},
		{"getMatrix",    l_getmatrix},
		{"setMatrix",    l_setmatrix},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.dipole");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_dip_new},
		{"help",                l_dip_help},
		{"metatable",           l_dip_mt},
		{NULL, NULL}
	};
		
	luaL_register(L, "Dipole", functions);
	lua_pop(L,1);

	Factory_registerItem(ENCODE_DIPOLE, newThing, lua_pushDipole, "Dipole");
	
}

extern "C"
{
DIPOLE_API int lib_register(lua_State* L);
DIPOLE_API int lib_version(lua_State* L);
DIPOLE_API const char* lib_name(lua_State* L);
DIPOLE_API int lib_main(lua_State* L, int argc, char** argv);
}

DIPOLE_API int lib_register(lua_State* L)
{
	registerDipole(L);
	return 0;
}

DIPOLE_API int lib_version(lua_State* L)
{
	return __revi;
}


DIPOLE_API const char* lib_name(lua_State* L)
{
	return "Dipole";
}

DIPOLE_API int lib_main(lua_State* L, int argc, char** argv)
{
	return 0;
}


