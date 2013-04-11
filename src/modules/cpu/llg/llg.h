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
#include <string>
#include "luabaseobject.h"

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef LLG_EXPORTS
  #define LLG_API __declspec(dllexport)
 #else
  #define LLG_API __declspec(dllimport)
 #endif
#else
 #define LLG_API 
#endif

#include <vector>
using namespace std;

class SpinSystem;

class LLG_API LLG : public LuaBaseObject
{
public:
	LLG(int encode_type=hash32("LLG.Base"));
	virtual ~LLG();
	
	LINEAGE1("LLG.Base")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	
	virtual bool apply(SpinSystem* spinfrom, double scaledmdt, SpinSystem* dmdt, SpinSystem* spinto, bool advancetime) {return true;}
	virtual bool apply(SpinSystem** spinfrom, double scaledmdt, SpinSystem** dmdt, SpinSystem** spinto, bool advancetime, int n);
	void fakeStep(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime);
		
	bool disableRenormalization;
	bool thermalOnlyFirstTerm;

	// 	void encode(buffer* b);
// 	int  decode(buffer* b);
};

#endif
