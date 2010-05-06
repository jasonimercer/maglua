#include "luacommon.h"
#include "main.h"

#ifndef LLGALIGNDEF
#define LLGALIGNDEF

#include "llg.h"

//make spin align with field
class LLGAlign : public LLG
{
public:
	LLGAlign();

	bool apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime);
};

#endif
