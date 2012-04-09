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

#include "hybridtaus.h"
#include <stdlib.h>


HybridTaus::HybridTaus()
	: RNG()
{
	nx = 4; 
	ny = 4;
	nz = 1;
	cpu_rng = 0;
	d_state = 0;
	seed();
}

HybridTaus::~HybridTaus()
{
	deinit();
	luaT_dec<RNG>(cpu_rng);
}
	
int HybridTaus::luaInit(lua_State* L)
{
	init();
	RNG::luaInit(L);
	if(luaT_is<RNG>(L, -1))
	{
		RNG* c = luaT_to<RNG>(L, -1);
		luaT_inc<RNG>(c);
		if(cpu_rng)
			luaT_dec<RNG>(cpu_rng);
		cpu_rng = c;
		cpu_rng->luaInit(L);
	}
	return 0;
}

void HybridTaus::init()
{
	if(d_state)
		return;

	twiddle = 1;

	HybridTausAllocState(&d_state, nx, ny, nz);
	HybridTausAllocRNG(&d_rngs, nx, ny, nz);
	
	seed();
}

void HybridTaus::deinit()
{
	if(!d_state)
		return;
	HybridTausFreeState(d_state);
	HybridTausFreeRNG(d_rngs);
	d_state = 0;
}

void HybridTaus::seed( const uint32 oneSeed )
{
	if(!d_state)
		init();
	
	_seed = oneSeed;

	HybridTausSeed(d_state, nx, ny, nz, _seed);
	
	if(cpu_rng)
	{
		cpu_rng->seed(oneSeed);
	}
}

uint32 HybridTaus::randInt()
{
	static long dumbRNG = 0;
	if(cpu_rng)
	{
		return cpu_rng->randInt();
	}
	if(dumbRNG >= 4294967296)
	{
		dumbRNG = 0;
	}
	
	return ++dumbRNG;
}


void HybridTaus::seed()
{
	// First try getting an array from /dev/urandom
	uint32 s = time(0);
#ifndef WIN32
	FILE* urandom = fopen( "/dev/urandom", "rb" );
	if( urandom )
	{
		if(fread(&s, sizeof(s), 1, urandom)) {} //to get around warning
		fclose(urandom);
	}
#endif
	seed( time(0) );
}

int HybridTaus::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Generates random variables on the GPU using the HybridTaus RNG.");
		lua_pushstring(L, "Optional 1 Number, Optional 1 RNG Object: The number is the seed. The RNG Object is a CPU based random number generator used for local CPU operations."); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
		
	return RNG::help(L);
}

float* HybridTaus::get6Normals(int _nx, int _ny, int _nz, int& t)
{
	if(_nx > nx || _ny > ny || _nz > nz)
	{
		deinit();
		nx = _nx;
		ny = _ny;
		nz = _nz;
		seed(_seed);
	}

	// flip fliop
	twiddle++;
	twiddle &= 0x1;

	if(!twiddle)
	{
		HybridTaus_get6Normals(d_state, d_rngs, nx, ny, nz);
	}

	t = twiddle;
	return d_rngs;
}




#include "info.h"
extern "C"
{
RANDOMCUDA_API int lib_register(lua_State* L);
RANDOMCUDA_API int lib_version(lua_State* L);
RANDOMCUDA_API const char* lib_name(lua_State* L);
RANDOMCUDA_API int lib_main(lua_State* L);
}

RANDOMCUDA_API int lib_register(lua_State* L)
{
	luaT_register<HybridTaus>(L);
	return 0;
}

RANDOMCUDA_API int lib_version(lua_State* L)
{
	return __revi;
}

RANDOMCUDA_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Random-Cuda32";
#else
	return "Random-Cuda32-Debug";
#endif
}

RANDOMCUDA_API int lib_main(lua_State* L)
{
	return 0;
}

