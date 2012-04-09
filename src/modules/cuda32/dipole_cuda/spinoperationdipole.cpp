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
#include "spinsystem.hpp"
#include "dipolesupport.h"
#include "info.h"

#include <stdlib.h>
#include <math.h>
#ifndef WIN32
#include <strings.h>
#endif

DipoleCuda::DipoleCuda(int nx, int ny, int nz)
	: LongRangeCuda("DipoleCuda", DIPOLE_SLOT, nx, ny, nz, ENCODE_DIPOLE)
{
}

int DipoleCuda::luaInit(lua_State* L)
{
    return LongRangeCuda::luaInit(L);
}

void DipoleCuda::push(lua_State* L)
{
    luaT_push<DipoleCuda>(L, this);
}


void DipoleCuda::encode(buffer* b)
{
	LongRangeCuda::encode(b);
}

int  DipoleCuda::decode(buffer* b)
{
	deinit();
	LongRangeCuda::decode(b);
	return 0;
}

DipoleCuda::~DipoleCuda()
{
	deinit();
}


void DipoleCuda::loadMatrixFunction(double* XX, double* XY, double* XZ, double* YY, double* YZ, double* ZZ)
{
	dipoleLoad(
		nx, ny, nz,
		gmax, ABC,
		XX, XY, XZ,
		YY, YZ, ZZ);
}









int DipoleCuda::help(lua_State* L)
{
    if(lua_gettop(L) == 0)
    {
        lua_pushstring(L, "Calculates the dipolar field of a *SpinSystem*");
        lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size");
        lua_pushstring(L, ""); //output, empty
        return 3;
    }

    return LongRangeCuda::help(L);
}

const luaL_Reg* DipoleCuda::luaMethods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
    if(m[127].name)return m;

    merge_luaL_Reg(m, LongRangeCuda::luaMethods());
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
DIPOLECUDA_API int lib_register(lua_State* L);
DIPOLECUDA_API int lib_version(lua_State* L);
DIPOLECUDA_API const char* lib_name(lua_State* L);
DIPOLECUDA_API int lib_main(lua_State* L);
}

int lib_register(lua_State* L)
{
	luaT_register<DipoleCuda>(L);
	return 0;
}

int lib_version(lua_State* L)
{
	return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Dipole-Cuda32";
#else
	return "Dipole-Cuda32-Debug";
#endif
}

int lib_main(lua_State* L)
{
	return 0;
}

