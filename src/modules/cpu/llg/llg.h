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
#include "luacommon.h"
#include <string>
#include "encodable.h"

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

class SpinSystem;

class LLG_API LLG : public Encodable
{
public:
	LLG(const char* llgtype, int etype);
	virtual ~LLG();
	
	virtual bool apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime) = 0;
	void fakeStep(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime);
	
	std::string type;
	
	int refcount;

	bool disablePrecession;

	void encode(buffer* b);
	int  decode(buffer* b);
};


LLG_API void registerLLG(lua_State* L);
LLG_API LLG* checkLLG(lua_State* L, int idx);
LLG_API void lua_pushLLG(lua_State* L, Encodable* llg);
LLG_API int  lua_isllg(lua_State* L, int idx);

#endif
