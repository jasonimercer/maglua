#include "isaac.h"
//#include <time.h>

Isaac::Isaac()
	:RNG()
{
	RNG::seed();
}

void Isaac::seed( const uint32 oneSeed )
{
	qtisaac.srand(oneSeed, oneSeed+1, oneSeed+2);

}

uint32 Isaac::randInt()                     // integer in [0,2^32-1]
{
	uint32 t = 0xFFFFFFFF & qtisaac.rand();
	return t;
}

int Isaac::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushfstring(L, "%s generates random variables using %s RNG.", Isaac::slineage(0), "Bob Jenkins's");
		lua_pushstring(L, ""); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
		
	return RNG::help(L);
}
