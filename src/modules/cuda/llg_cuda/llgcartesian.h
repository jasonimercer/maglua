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

#ifndef LLGCARTESIAN
#define LLGCARTESIAN

#include "llg.h"

class SpinSystem;

class LLGCartesian : public LLG
{
public:
	LLGCartesian();
	~LLGCartesian();

	LINEAGE2("LLG.Cartesian", "LLG.Base")
	static const luaL_Reg* luaMethods() {return LLG::luaMethods();}
	static int help(lua_State* L);

	bool apply(SpinSystem* spinfrom, double scaledmdt, SpinSystem* dmdt, SpinSystem* spinto, bool advancetime);
	bool apply(SpinSystem** spinfrom, double scaledmdt, SpinSystem** dmdt, SpinSystem** spinto, bool advancetime, int n);

};

#endif
