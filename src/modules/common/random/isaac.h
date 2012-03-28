/*
------------------------------------------------------------------------------
rand.h: definitions for a random number generator
By Bob Jenkins, 1996, Public Domain
MODIFIED:
  960327: Creation (addition of randinit, really)
  970719: use context, not global variables, for internal state
  980324: renamed seed to flag
  980605: recommend RANDSIZL=4 for noncryptography.
  010626: note this is public domain
------------------------------------------------------------------------------
*/
#ifndef ISAAC_RNG
#define ISAAC_RNG
#define RANDSIZL   (8)
#define RANDSIZ    (1<<RANDSIZL)

#ifndef STANDARD_H
#include "standard.h"
#endif

#include "random.h"

/* context of random number generator */
typedef struct randctx
{
  ub4 randcnt;
  ub4 randrsl[RANDSIZ];
  ub4 randmem[RANDSIZ];
  ub4 randa;
  ub4 randb;
  ub4 randc;
} randctx;

class RANDOM_API Isaac : public RNG
{
public:
	Isaac();
	
	LINEAGE2("Random.Isaac", "Random.Base")
	static const luaL_Reg* luaMethods() {return RNG::luaMethods();}
	virtual int luaInit(lua_State* L) {return RNG::luaInit(L);}
	virtual void push(lua_State* L) {luaT_push<Isaac>(L, this);}
	static int help(lua_State* L);

	
	uint32 randInt();                     // integer in [0,2^32-1]
	uint32 randInt( const uint32 n );     // integer in [0,n] for n < 2^32

	void seed( const uint32 oneSeed );
	void seed( ub4* const bigSeed, const uint32 seedLength );
	void seed(); //seed by time

private:
	struct randctx ctx;

	ub4 *pNext;     // next value to get from state
	int left;       // number of values left before reload needed
	
};

#endif
