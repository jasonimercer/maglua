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
	: LongRange(Dipole::typeName(), DIPOLE_SLOT, nx, ny, nz, hash32(Dipole::typeName()))
{

}

int Dipole::luaInit(lua_State* L)
{
	return LongRange::luaInit(L); //gets nx, ny, nz, nxyz
}

void Dipole::push(lua_State* L)
{
	luaT_push<Dipole>(L, this);
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




int Dipole::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Calculates the dipolar field of a *SpinSystem*");
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	return LongRange::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Dipole::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LongRange::luaMethods());
	static const luaL_Reg _m[] =
	{
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}



extern "C"
{
DIPOLE_API int lib_register(lua_State* L);
DIPOLE_API int lib_version(lua_State* L);
DIPOLE_API const char* lib_name(lua_State* L);
DIPOLE_API int lib_main(lua_State* L);
}

DIPOLE_API int lib_register(lua_State* L)
{
	luaT_register<Dipole>(L);
	return 0;
}

DIPOLE_API int lib_version(lua_State* L)
{
	return __revi;
}


DIPOLE_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Dipole";
#else
	return "Dipole-Debug";
#endif
}

DIPOLE_API int lib_main(lua_State* L)
{
	return 0;
}


