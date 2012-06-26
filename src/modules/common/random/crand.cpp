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

#include "crand.h"

#include <stdlib.h>


#ifdef WIN32
static int (*globalRand)() = rand;
#endif

CRand::CRand()
	: RNG()
{
	RNG::seed();
}

void CRand::seed( const uint32 oneSeed )
{
	_seed = oneSeed;
#ifndef WIN32
	srand(_seed);
#endif
}

uint32 CRand::randInt()
{
#ifndef WIN32
	uint32 t = 0xFFFFFFFF & (rand_r(&_seed) ^ (rand_r(&_seed) << 16));
#else
	uint32 t = 0xFFFFFFFF & (globalRand() ^ (globalRand() << 16));
#endif
	return t;
}

#if 0
void CRand::seed()
{
	// First try getting an array from /dev/urandom
#ifndef WIN32
	FILE* urandom = fopen( "/dev/urandom", "rb" );
	if( urandom )
	{
		if(fread(&_seed, sizeof(_seed), 1, urandom))
		{
			fclose(urandom);
			return;
		}
		fclose(urandom);
	}
#endif
	seed( time(0) );
}
#endif

int CRand::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushfstring(L, "%s generates random variables using the %s RNG.", CRand::slineage(0), "C Library");
		lua_pushstring(L, ""); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
		
	return RNG::help(L);
}


