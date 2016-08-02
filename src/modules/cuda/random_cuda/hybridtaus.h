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

#include "modules/common/random/random.h"
#include "hybridtaus.hpp"

#ifndef HYBRIDTAUS_CLASS_H
#define HYBRIDTAUS_CLASS_H

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)
 #ifdef RANDOMCUDA_EXPORTS
  #define RANDOMCUDA_API __declspec(dllexport)
 #else
  #define RANDOMCUDA_API __declspec(dllimport)
 #endif
#else
 #define RANDOMCUDA_API 
#endif

class RANDOMCUDA_API HybridTaus : public RNG
{
public:
	HybridTaus();
	~HybridTaus();
	
	LINEAGE2("Random.HybridTaus", "Random.Base")
	static const luaL_Reg* luaMethods() {return RNG::luaMethods();}
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);

	float* get6Normals(int nx, int ny, int nz, int& twiddle);

	
	uint32 randInt();                     // integer in [0,2^32-1]
	
	void seed( const uint32 oneSeed );
	void seed(); //seed by time

	void init();
	void deinit();
	
	state_t* d_state;
	float* d_rngs;
private:
	unsigned int _seed;
	int twiddle;
	
	int nx, ny, nz; //size of state
	
	RNG* cpu_rng;
};


#endif

