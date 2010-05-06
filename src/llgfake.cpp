#include <math.h>
#include "llgfake.h"
#include "spinsystem.h"
#include "spinoperation.h"

LLGFake::LLGFake()
	: LLG("Fake", ENCODE_LLGFAKE)
{
}


bool LLGFake::apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime)
{
	const double dt    = spinfrom->dt;
	
	if(advancetime)
		spinto->time = spinfrom->time + dt;
	return true;
}

