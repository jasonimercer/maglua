#ifndef LLGCARTESIAN
#define LLGCARTESIAN

#include "luacommon.h"
#include "llg.h"
#include <string>

class SpinSystem;

class LLGCartesian : public LLG
{
public:
	LLGCartesian();

	bool apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto);
	
};

#endif
