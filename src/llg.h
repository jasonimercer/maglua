#ifndef LLGDEF
#define LLGDEF
#include "luacommon.h"
#include <string>
#include "encodable.h"

class SpinSystem;

class LLG : public Encodable
{
public:
	LLG(const char* llgtype, int etype);
	virtual ~LLG();
	
	virtual bool apply(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime) = 0;
	void fakeStep(SpinSystem* spinfrom, SpinSystem* fieldfrom, SpinSystem* spinto, bool advancetime);
	
	std::string type;
	
	int refcount;

	bool disablePrecession;

	void encode(buffer* b) const;
	int  decode(buffer* b);
};


void registerLLG(lua_State* L);
LLG* checkLLG(lua_State* L, int idx);
void lua_pushLLG(lua_State* L, LLG* llg);
int  lua_isllg(lua_State* L, int idx);

#endif
