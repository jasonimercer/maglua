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

#include <math.h>
#include "llgfake.h"
#include "spinsystem.h"
#include "spinoperation.h"

LLGFake::LLGFake()
	: LLG(hash32(LLGFake::typeName()))
{
}


bool LLGFake::apply(SpinSystem* spinfrom, double scaledmdt, SpinSystem* dmdt, SpinSystem* spinto, bool advancetime)
// bool LLGFake::apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime)
{
	const double dt    =  scaledmdt * dmdt->dt;
	
	if(advancetime)
		spinto->time = spinfrom->time + dt;
	return true;
}

int LLGFake::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "LLG.Fake advances a *SpinSystem* through time without changing spin orientation");
		lua_pushstring(L, ""); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
		
	return LLG::help(L);
}
