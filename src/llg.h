#ifndef LLGDEF
#define LLGDEF
#include "luacommon.h"
#include <string>

class SpinSystem;

class LLG
{
public:
	LLG(const char* llgtype);
	virtual ~LLG();
	
	virtual bool apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto) = 0;
	void fakeStep(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto);
	
	double alpha;
	double dt;
	double gamma;
	std::string type;
	
	int refcount;
};


void registerLLG(lua_State* L);
LLG* checkLLG(lua_State* L, int idx);
int  lua_isllg(lua_State* L, int idx);

#endif
