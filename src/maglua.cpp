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

extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
        
int lib_register(lua_State* L);
int lib_deps(lua_State* L);
}

#include "spinsystem.h"
#include "luacommon.h"
#include "spinoperation.h"
#include "llg.h"
#include "llgquat.h"
#include "spinoperationexchange.h"
#include "spinoperationappliedfield.h"
#include "spinoperationanisotropy.h"
#include "spinoperationdipole.h"
#include "mersennetwister.h"
#include "spinoperationthermal.h"
#include "interpolatingfunction.h"
#include "interpolatingfunction2d.h"
#include "luampi.h"
	
int lib_register(lua_State* L)
{
	registerSpinSystem(L);
	registerLLG(L);
	registerExchange(L);
	registerAppliedField(L);
	registerAnisotropy(L);
	registerDipole(L);
	registerRandom(L);
	registerThermal(L);
	registerInterpolatingFunction(L);
	registerInterpolatingFunction2D(L);

#ifdef _MPI
	registerMPI(L);
#endif

	return 0;
}

int lib_deps(lua_State* L)
{
	return 0;
}
