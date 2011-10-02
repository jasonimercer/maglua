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

CRand::CRand()
	: RNG("CRandom")
{
	seed();
}

void CRand::seed( const uint32 oneSeed )
{
	_seed = oneSeed;
}

uint32 CRand::randInt()
{
	uint32 t = 0xFFFFFFFF & (rand_r(&_seed) ^ (rand_r(&_seed) << 16));
	return t;
}


void CRand::seed()
{
	// First try getting an array from /dev/urandom
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
	
	seed( time(0) );
}
