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

#ifndef RANDOMBASE_HPP
#define RANDOMBASE_HPP

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#include <stdint.h>
#include <time.h>
#include <ctype.h>

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef RANDOM_EXPORTS
  #define RANDOM_API __declspec(dllexport)
 #else
  #define RANDOM_API __declspec(dllimport)
 #endif
#else
 #define RANDOM_API 
#endif

#include "luabaseobject.h"
#include <string>
using namespace std;
typedef unsigned long uint32;  // unsigned integer type, at least 32 bits

class RANDOM_API RNG : public LuaBaseObject
{
public:
	RNG();
	virtual ~RNG() {};
	
	LINEAGE1("Random.Base")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	static int help(lua_State* L);

	void encode(buffer* b) {}
	int  decode(buffer* b) {return 0;}

	virtual uint32 randInt() {return 0;}                 // integer in [0,2^32-1]
	virtual double rand();                        // real number in [0,1]
	virtual double rand( const double n );        // real number in [0,n]
	virtual double randExc();                     // real number in [0,1)
	virtual double randExc( const double n );     // real number in [0,n)
	virtual double randDblExc();                  // real number in (0,1)
	virtual double randDblExc( const double n );  // real number in (0,n)

	// Access to nonuniform random number distributions
	virtual double randNorm( const double mean = 0.0, const double stddev = 1.0 );
	
	virtual void seed( const uint32 oneSeed ) {};
	virtual void seed(); //seed by time

protected:
	double gaussPair[2];
	unsigned char __gaussStep;
};

#endif
