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

#ifndef LLGQUATDEF
#define LLGQUATDEF

#include "llg.h"

class LLGQuaternion : public LLG
{
public:
	LLGQuaternion();
	~LLGQuaternion();
	LINEAGE2("LLG.Quaternion", "LLG.Base")
	static const luaL_Reg* luaMethods() {return LLG::luaMethods();}
	static int help(lua_State* L);

	virtual bool apply(SpinSystem*  spinfrom, double scaledmdt, SpinSystem*  dmdt, SpinSystem*  spinto, bool advancetime);
	virtual bool apply(SpinSystem** spinfrom, double scaledmdt, SpinSystem** dmdt, SpinSystem** spinto, bool advancetime, int n);
};

#endif
