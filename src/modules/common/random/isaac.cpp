#include "isaac.h"
//#include <time.h>

Isaac::Isaac()
	:RNG()
{
	seed();
}

void Isaac::seed( const uint32 oneSeed )
{
	qtisaac.srand(oneSeed, oneSeed+1, oneSeed+2);

}

void Isaac::seed()
{
	// Seed the generator with an array from /dev/urandom if available
	// Otherwise use a hash of time() and clock() values
	
	// First try getting an array from /dev/urandom
	FILE* urandom = fopen( "/dev/urandom", "rb" );
	if( urandom )
	{
		ISAAC_INT v[3];
		bool success = true;
		success = fread(v, sizeof(ISAAC_INT), 3, urandom );
		fclose(urandom);
		if(success)
		{
			qtisaac.srand(v[0], v[1], v[2]);
			return;
		}
	}
	
	//seed( time(0) );
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
