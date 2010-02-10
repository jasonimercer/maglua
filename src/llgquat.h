#include "luacommon.h"
#include "main.h"

#ifndef LLGQUATDEF
#define LLGQUATDEF

#include "llg.h"

class LLGQuaternion : public LLG
{
public:
	LLGQuaternion();

	bool apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto);
};

#endif
