#ifndef LLGFAKE
#define LLGFAKE

#include "luacommon.h"
#include "llg.h"
#include <string>

class SpinSystem;

class LLGFake : public LLG
{
public:
	LLGFake();

	bool apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto);
};

#endif
