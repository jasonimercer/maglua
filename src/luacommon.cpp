/******************************************************************************
* Copyright (C) 2008-2010 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#include "main.h"
#include "luacommon.h"

void registerLibs(lua_State* L)
{
	luaL_openlibs(L);
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
}
