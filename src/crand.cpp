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
	uint32 t = 0xFFFFFFFF & rand_r(&_seed);
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
