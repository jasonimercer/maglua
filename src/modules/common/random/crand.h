/******************************************************************************
* Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#include <stdlib.h>
#include "random.h"

#ifndef CRANDOM_HPP
#define CRANDOM_HPP

class RANDOM_API CRand : public RNG
{
public:
	CRand();
	
	LINEAGE2("Random.CRand", "Random.Base")
	static const luaL_Reg* luaMethods() {return RNG::luaMethods();}
	virtual int luaInit(lua_State* L) {return RNG::luaInit(L);}
	virtual void push(lua_State* L) {luaT_push<CRand>(L, this);}
	static int help(lua_State* L);
	
	uint32 randInt();                     // integer in [0,2^32-1]
	
	void seed( const uint32 oneSeed );

private:
	unsigned int _seed;
};


#endif

