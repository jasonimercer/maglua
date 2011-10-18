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

class SpinSystem;

class CORE_API LLG : public Encodable
{
public:
	LLG(const char* llgtype, int etype);
	virtual ~LLG();
	
	virtual bool apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime) = 0;
	void fakeStep(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime);
	
	std::string type;
	
	int refcount;

	bool disablePrecession;

	void encode(buffer* b) const;
	int  decode(buffer* b);
};


CORE_API void registerLLG(lua_State* L);
CORE_API LLG* checkLLG(lua_State* L, int idx);
CORE_API void lua_pushLLG(lua_State* L, LLG* llg);
CORE_API int  lua_isllg(lua_State* L, int idx);

#endif
