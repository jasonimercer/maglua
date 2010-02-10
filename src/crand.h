#include <stdlib.h>
#include "random.h"

#ifndef CRANDOM_HPP
#define CRANDOM_HPP

class CRand : public RNG
{
public:
	CRand();
	
	uint32 randInt();                     // integer in [0,2^32-1]
	
	void seed( const uint32 oneSeed );
	void seed(); //seed by time

private:
	unsigned int _seed;
};


#endif

