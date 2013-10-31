/******************************************************************************
* Copyright (C) 2008-2013 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#ifndef KMCDEF
#define KMCDEF

#include "maglua.h"
#include <string>
#include "luabaseobject.h"

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef KMC_EXPORTS
  #define KMC_API __declspec(dllexport)
 #else
  #define KMC_API __declspec(dllimport)
 #endif
#else
 #define KMC_API 
#endif

#include <vector>
using namespace std;

class KMC_API KMC : public LuaBaseObject
{
public:
	KMC();
	virtual ~KMC();
	
	LINEAGE1("KMC")

	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);
	
	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	
	int   getInternalData(lua_State* L);
	void  setInternalData(lua_State* L, int stack_pos);
	void  deinit();

	int data_ref;
};

#endif
