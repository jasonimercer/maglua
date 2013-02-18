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

#ifndef SPINOPERATIONEXCHANGE
#define SPINOPERATIONEXCHANGE

#include "spinoperation.h"

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef EXCHANGE_EXPORTS
  #define EXCHANGE_API __declspec(dllexport)
 #else
  #define EXCHANGE_API __declspec(dllimport)
 #endif
#else
 #define EXCHANGE_API 
#endif

class EXCHANGE_API Exchange : public SpinOperation
{
public:
	Exchange(int nx=32, int ny=32, int nz=1);
	virtual ~Exchange();
	
	LINEAGE2("Exchange", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	static int help(lua_State* L);

	bool apply(SpinSystem* ss);

	void addPath(int site1, int site2, double strength);

	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	void opt();

	typedef struct sss
	{
		int fromsite;
		int tosite;
		double strength;
	} sss;

	
	int numPaths() {return num;}
	bool getPath(int idx, int& fx, int& fy, int& fz, int& tx, int& ty, int& tz, double& strength);
	int mergePaths();
// private:
	void deinit();
	
	int size;
	int num;
	sss* pathways;
};

EXCHANGE_API Exchange* checkExchange(lua_State* L, int idx);
EXCHANGE_API void registerExchange(lua_State* L);
EXCHANGE_API void lua_pushExchange(lua_State* L, LuaBaseObject* ex);

#endif
