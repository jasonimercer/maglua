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

#ifndef LLGDEF
#define LLGDEF
#include "maglua.h"
#include "luabaseobject.h"

class SpinSystem;

class LLG : public LuaBaseObject
{
public:
	LLG(int encode_type=hash32("LLG.Base"));
	virtual ~LLG();
	
	LINEAGE1("LLG.Base")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	
	virtual bool apply(SpinSystem*  spinfrom, double scaledmdt, SpinSystem*  dmdt, SpinSystem*  spinto, bool advancetime)  {return true;}
	// multiple data version
	virtual bool apply(SpinSystem** spinfrom, double scaledmdt, SpinSystem** dmdt, SpinSystem** spinto, bool advancetime, int n)  {return true;}
	void fakeStep(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime);
	
	bool disableRenormalization;
	bool thermalOnlyFirstTerm;
	
	void encode(buffer* b);
	int  decode(buffer* b);
};

#endif
