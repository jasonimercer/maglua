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



#ifndef SPINOPERATIONMULTIPOLE
#define SPINOPERATIONMULTIPOLE

#ifdef WIN32
 #ifdef MULTIPOLE_EXPORTS
  #define MULTIPOLE_API __declspec(dllexport)
 #else
  #define MULTIPOLE_API __declspec(dllimport)
 #endif
#else
 #define MULTIPOLE_API 
#endif

#if 0

#include "spinoperation.h"
#include "array.h"
#include "octtree.h"

using namespace std;

class MULTIPOLE_API Multipole : public SpinOperation
{
public:
	Multipole(int nx=4, int ny=4, int nz=1);
	virtual ~Multipole();
	
	LINEAGE2("Multipole", "SpinOperation")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	static int help(lua_State* L);

	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	void precompute();
	
	void init();
	void deinit();
	
	bool apply(SpinSystem* ss);
	
	dArray* x;
	dArray* y;
	dArray* z;
	dArray* weight;
	
	OctTree* oct;
};


#endif
#endif
