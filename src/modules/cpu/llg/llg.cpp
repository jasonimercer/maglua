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

#include "llg.h"
#include "llgcartesian.h"
#include "llgquat.h"
#include "llgfake.h"
#include "llgalign.h"
#include <string.h>
#include "spinsystem.h"

LLG::LLG(int encode_type)
: LuaBaseObject(encode_type), disablePrecession(false)
{
	
}

int LLG::luaInit(lua_State* L)
{
	return 0;
}

// void LLG::encode(buffer* b)
// {
// 	int len = type.length()+1;
// 	encodeInteger( len, b);
// 	encodeBuffer(type.c_str(), len, b);
// }
// 
// int  LLG::decode(buffer* b)
// {
// 	int len = decodeInteger(b);
// 	
// 	char* t = new char[len];
// 	decodeBuffer(t, len, b);
// 	
// 	type = t;
// 	delete [] t;
// 	
// 	return 0;
// }


LLG::~LLG()
{
	
}


/*
LLG* checkLLG(lua_State* L, int idx)
{
	LLG** pp = (LLG**)luaL_checkudata(L, idx, "MERCER.llg");
	luaL_argcheck(L, pp != NULL, 1, "`LLG' expected");
	return *pp;
}

int  lua_isllg(lua_State* L, int idx)
{
	LLG** pp = (LLG**)luaL_checkudata(L, idx, "MERCER.llg");
	return pp != 0;
}


void lua_pushLLG(lua_State* L, Encodable* )
{
	LLG* llg = dynamic_cast<LLG*>();
	if(!llg) return;
	llg->refcount++;
	
	LLG** pp = (LLG**)lua_newuserdata(L, sizeof(LLG**));
	
	*pp = llg;
	luaL_getmetatable(L, "MERCER.llg");
	lua_setmetatable(L, -2);
}


int l_new(lua_State* L)
{
	LLG* llg = 0;
	
	if(strcasecmp(lua_tostring(L, 1), "Cartesian") == 0)
	{
		llg = new LLGCartesian;
	}
	else if(strcasecmp(lua_tostring(L, 1), "Quaternion") == 0)
	{
		llg = new LLGQuaternion;
	}
	else if(strcasecmp(lua_tostring(L, 1), "Fake") == 0)
	{
		llg = new LLGFake;
	}
	else if(strcasecmp(lua_tostring(L, 1), "Align") == 0)
	{
		llg = new LLGAlign;
	}
	if(!llg)
		return luaL_error(L, "Unknown LLG type `%s'", lua_tostring(L, 1));
	
	lua_pushLLG(L, llg);
	
	return 1;
}
*/



int l_gettype(lua_State* L)
{
	LUA_PREAMBLE(LLG, llg, 1);
	lua_pushstring(L, llg->lineage(0));
	return 1;
}



// static int l_mt(lua_State* L)
// {
// 	luaL_getmetatable(L, "MERCER.llg");
// 	return 1;
// }





// 
// The apply function is highly overloaded 
//  it looks like this
//  ll:apply(A, x, B, C, yn)
// 
// Where 
// 
// A = "from system", this "old system" in "x_new = x_old + dt * dx/dt"
// x = optional scalar, default = 1. Scales the dt.
// B = system that is used to calculate dx/dt. Also the souce of the timestep
// C = "to system". This is where the result will get stored
// yn = boolean, optional. default = true. If true, the time of x_new is updated, else, not updated.
static int l_apply(lua_State* L)
{
	LUA_PREAMBLE(LLG, llg, 1);
				
	bool advanceTime = true;
	double scale_dmdt = 1.0;
	
	if(lua_isboolean(L, -1))
		if(lua_toboolean(L, -1) == 0)
			advanceTime = false;
		
	int sys1_pos = 2; //because the llg operator is at 1
	int sys2_pos = 3;
	if(lua_isnumber(L, 3))
	{
		scale_dmdt = lua_tonumber(L, 3);
		sys2_pos = 4; //need to shift it
	}
	
	SpinSystem* ss[3]; //from, dmdt, to
	
	ss[0] = luaT_to<SpinSystem>(L, sys1_pos);
	ss[1] = ss[0];
	ss[2] = ss[0];
	
	SpinSystem* t;
	//get 2nd and 3rd systems if present
	for(int i=0; i<2; i++)
	{
		if(luaT_is<SpinSystem>(L, sys2_pos+i))
			ss[1+i] = luaT_to<SpinSystem>(L, sys2_pos+i);
	}	

	if(!llg || !ss[0])
	{
		return luaL_error(L, "apply requires 1, 2 or 3 spin systems (extra boolean argument allowed to control timestep, optional 2nd arg can scale timestep)");
	}

	llg->apply(ss[0], scale_dmdt, ss[1], ss[2], advanceTime);
	
	return 0;
}


static int l_type(lua_State* L)
{
	LUA_PREAMBLE(LLG, llg, 1);
	lua_pushstring(L, llg->lineage(0));
	return 1;
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* LLG::luaMethods()
{
	if(m[127].name)
		return m;

	static const luaL_Reg _m[] =
	{
		//{"__tostring",   l_tostring},
		{"apply",        l_apply},
		{"type",         l_type},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}


static int l_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "LLG advances a *SpinSystem* through time using a form of the LLG equation.");
		lua_pushstring(L, ""); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	if(lua_istable(L, 1))
	{
		return 0;
	}
	
	if(!lua_iscfunction(L, 1))
	{
		return luaL_error(L, "help expect zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
// 	if(func == l_new)
// 	{
// 		lua_pushstring(L, "Create a new LLG object.");
// 		lua_pushstring(L, "1 string: The string argument defines the LLG type. It may be one of the following:\n\"Cartesian\" - update the components of the spins indivifually.\n\"Quaternion\" - use rotation methods to update all components simultaneously.\n\"Fake\" - do nothing to spins, update the timestep.\n\"Align\" - align the spins with the local field.");
// 		lua_pushstring(L, "1 LLG object");
// 		return 3;
// 	}
	
	if(func == l_apply)
	{
		lua_pushstring(L, "Compute 1 LLG Euler Step.");
		lua_pushstring(L, "1 *SpinSystem*, Oprional Number, Optional 2 *SpinSystem*s, Optional Boolean: " 
		"Make 1 Euler step from 1st system using 2nd system to compute derivative (defaulting to 1st system), "
		"scaling timestep be optional number (default 1.0) and storing in 3rd system (defaulting to 1st system)."
		"If last argument is the boolean \"false\", the time will not be incremented");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_gettype)
	{
		lua_pushstring(L, "Determine which type of the LLG object.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 string: \"Cartesian\", \"Quaternion\", \"Fake\" or \"Align\"");
		return 3;
	}
	
	return 0;
}



#include "info.h"
extern "C"
{
LLG_API int lib_register(lua_State* L);
LLG_API int lib_version(lua_State* L);
LLG_API const char* lib_name(lua_State* L);
LLG_API int lib_main(lua_State* L);
}

LLG_API int lib_register(lua_State* L)
{
	luaT_register<LLG>(L);
	luaT_register<LLGCartesian>(L);
	luaT_register<LLGQuaternion>(L);
	return 0;
}

LLG_API int lib_version(lua_State* L)
{
	return __revi;
}

LLG_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "LLG";
#else
	return "LLG-Debug";
#endif
}

LLG_API int lib_main(lua_State* L)
{
	return 0;
}


