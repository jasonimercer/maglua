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
#include <ctype.h>
#include <time.h>

#define __ISAAC64 
#ifndef __ISAAC64
   typedef unsigned long int UINT32;
   const UINT32 GOLDEN_RATIO = UINT32(0x9e3779b9);
   typedef UINT32 ISAAC_INT;
#else   // __ISAAC64

//typedef __uint64_t UINT64;
typedef uint64_t UINT64;

const UINT64 GOLDEN_RATIO = UINT64(0x9e3779b97f4a7c13);
typedef UINT64 ISAAC_INT;
#endif  // __ISAAC64


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


#include <string>
using namespace std;
typedef unsigned long uint32;  // unsigned integer type, at least 32 bits

class RANDOM_API RNG
{
public:
	RNG(const char* name);
	virtual ~RNG() {};
	
	virtual uint32 randInt() = 0;                 // integer in [0,2^32-1]
	virtual double rand();                        // real number in [0,1]
	virtual double rand( const double n );        // real number in [0,n]
	virtual double randExc();                     // real number in [0,1)
	virtual double randExc( const double n );     // real number in [0,n)
	virtual double randDblExc();                  // real number in (0,1)
	virtual double randDblExc( const double n );  // real number in (0,n)

	// Access to nonuniform random number distributions
	virtual double randNorm( const double mean = 0.0, const double stddev = 1.0 );
	
	virtual void seed( const uint32 oneSeed ) = 0;
	virtual void seed() = 0; //seed by time

	string type;
protected:
	double gaussPair[2];
	unsigned char __gaussStep;
};


RANDOM_API RNG* checkRandom(lua_State* L, int idx);
RANDOM_API void registerRandom(lua_State* L);


#endif
