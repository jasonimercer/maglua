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

LLG::~LLG()
{
	
}








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


// static int l_type(lua_State* L)
// {
// 	LUA_PREAMBLE(LLG, llg, 1);
// 	lua_pushstring(L, llg->lineage(0));
// 	return 1;
// }



int LLG::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "LLG advances a *SpinSystem* through time using a form of the LLG equation. This is an abstract base class and is not intended to be used directly.");
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
	
// 	if(func == l_gettype)
// 	{
// 		lua_pushstring(L, "Determine which type of the LLG object.");
// 		lua_pushstring(L, "");
// 		lua_pushstring(L, "1 string: \"Cartesian\", \"Quaternion\", \"Fake\" or \"Align\"");
// 		return 3;
// 	}
	
	return LuaBaseObject::help(L);
}



static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* LLG::luaMethods()
{
	if(m[127].name)
		return m;

	static const luaL_Reg _m[] =
	{
		{"apply",        l_apply},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}



#include "info.h"
extern "C"
{
LLG_API int lib_register(lua_State* L);
LLG_API int lib_version(lua_State* L);
LLG_API const char* lib_name(lua_State* L);
LLG_API int lib_main(lua_State* L);
}

#include "llg_wrapper.h"
LLG_API int lib_register(lua_State* L)
{
	luaT_register<LLG>(L);
	luaT_register<LLGCartesian>(L);
	luaT_register<LLGQuaternion>(L);
	luaT_register<LLGAlign>(L);
	luaT_register<LLGFake>(L);	
	
	if(luaL_dostring(L, __llg_wrapper()))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
	}
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


