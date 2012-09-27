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

#ifndef WOODDEF
#define WOODDEF


#include "spinoperation.h"
#include "spinoperationanisotropy.h"

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef WOOD_EXPORTS
  #define WOOD_API __declspec(dllexport)
 #else
  #define WOOD_API __declspec(dllimport)
 #endif
#else
 #define WOOD_API 
#endif

class SpinSystem;

class WOOD_API Wood : public SpinOperation
{
public:
	Wood();
	virtual ~Wood();
	
	LINEAGE2("Wood", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	static int help(lua_State* L);
	
	virtual bool apply(SpinSystem* ss_src, Anisotropy* ani, SpinSystem* ss_dest, int& updates, int index);

	void encode(buffer* b);
	int  decode(buffer* b);
	
	
	void deinit();
	void init();
	
	double DN;
};


#endif
