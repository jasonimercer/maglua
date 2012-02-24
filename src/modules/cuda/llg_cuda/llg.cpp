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
// #include "llgfake.h"
// #include "llgalign.h"
#include <string.h>
#include "spinsystem.h"

LLG::LLG(const char* llgtype, int etype)
: Encodable(etype), type(llgtype), refcount(0), disablePrecession(false)
{
	
}

void LLG::encode(buffer* b)
{
	int len = type.length()+1;
	encodeInteger( len, b);
	encodeBuffer(type.c_str(), len, b);
}

int  LLG::decode(buffer* b)
{
	int len = decodeInteger(b);
	
	char* t = new char[len];
	decodeBuffer(t, len, b);
	
	type = t;
	delete [] t;
	
	return 0;
}


LLG::~LLG()
{
	
}



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


void lua_pushLLG(lua_State* L, LLG* llg)
{
	llg->refcount++;
	
	LLG** pp = (LLG**)lua_newuserdata(L, sizeof(LLG**));
	
	*pp = llg;
	luaL_getmetatable(L, "MERCER.llg");
	lua_setmetatable(L, -2);
}


int l_llg_new(lua_State* L)
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
// 		llg = new LLGFake;
	}
	else if(strcasecmp(lua_tostring(L, 1), "Align") == 0)
	{
// 		llg = new LLGAlign;
	}
	if(!llg)
		return luaL_error(L, "Unknown LLG type `%s'", lua_tostring(L, 1));
	
	lua_pushLLG(L, llg);
	
	return 1;
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


int l_llg_apply(lua_State* L)
{
	LLG* llg = checkLLG(L, 1);
				
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
	
	ss[0] = checkSpinSystem(L, sys1_pos);
	ss[1] = ss[0];
	ss[2] = ss[0];
	
	SpinSystem* t;
	//get 2nd and 3rd systems if present
	for(int i=0; i<2; i++)
	{
		if(lua_isSpinSystem(L, sys2_pos+i))
			ss[1+i] = lua_toSpinSystem(L, sys2_pos+i);
	}	

	if(!llg || !ss[0])
	{
		return luaL_error(L, "apply requires 1, 2 or 3 spin systems (extra boolean argument allowed to control timestep, optional 2nd arg can scale timestep)");
	}

	llg->apply(ss[0], scale_dmdt, ss[1], ss[2], advanceTime);
	
	return 0;
}



int l_llg_gettype(lua_State* L)
{
	LLG* llg = checkLLG(L, 1);
	if(!llg) return 0;
	lua_pushstring(L, llg->type.c_str());
	return 1;
}



int l_llg_gc(lua_State* L)
{
	LLG* llg = checkLLG(L, 1);
	if(!llg) return 0;
	
	llg->refcount--;
	if(llg->refcount == 0)
		delete llg;
	return 0;
}

int l_llg_tostring(lua_State* L)
{
	LLG* llg = checkLLG(L, 1);
	if(!llg) return 0;
	
	lua_pushfstring(L, "LLG(%s)", llg->type.c_str());
	return 1;
}


static int l_llg_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.llg");
	return 1;
}

static int l_llg_help(lua_State* L)
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
	
	if(func == l_llg_new)
	{
		lua_pushstring(L, "Create a new LLG object.");
		lua_pushstring(L, "1 string: The string argument defines the LLG type. It may be one of the following:\n\"Cartesian\" - update the components of the spins indivifually.\n\"Quaternion\" - use rotation methods to update all components simultaneously.\n\"Fake\" - do nothing to spins, update the timestep.\n\"Align\" - align the spins with the local field.");
		lua_pushstring(L, "1 LLG object");
		return 3;
	}
	
	if(func == l_llg_apply)
	{
		lua_pushstring(L, "Compute 1 LLG Euler Step.");
		lua_pushstring(L, "1 *SpinSystem*, Oprional Number, Optional 2 *SpinSystem*s, Optional Boolean: " 
		"Make 1 Euler step from 1st system using 2nd system to compute derivative (defaulting to 1st system), "
		"scaling timestep be optional number (default 1.0) and storing in 3rd system (defaulting to 1st system)."
		"If last argument is the boolean \"false\", the time will not be incremented");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_llg_gettype)
	{
		lua_pushstring(L, "Determine which type of the LLG object.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 string: \"Cartesian\", \"Quaternion\", \"Fake\" or \"Align\"");
		return 3;
	}
	
	return 0;
}



void registerLLG(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
	{"__gc",         l_llg_gc},
	{"__tostring",   l_llg_tostring},
	{"apply",        l_llg_apply},
	{"type",         l_llg_gettype},
	{NULL, NULL}
	};
	
	luaL_newmetatable(L, "MERCER.llg");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
	
	static const struct luaL_reg functions [] = {
		{"new",                 l_llg_new},
		{"help",                l_llg_help},
		{"metatable",           l_llg_mt},
		{NULL, NULL}
	};
	
	luaL_register(L, "LLG", functions);
	lua_pop(L,1);	
}

#include "info.h"
extern "C"
{
LLGCUDA_API int lib_register(lua_State* L);
LLGCUDA_API int lib_version(lua_State* L);
LLGCUDA_API const char* lib_name(lua_State* L);
LLGCUDA_API int lib_main(lua_State* L);
}

LLGCUDA_API int lib_register(lua_State* L)
{
	registerLLG(L);
	return 0;
}

LLGCUDA_API int lib_version(lua_State* L)
{
	return __revi;
}

LLGCUDA_API const char* lib_name(lua_State* L)
{
#ifdef NDEBUG 
	return "LLG-Cuda";
#else
	return "LLG-Cuda-Debug";
#endif
}

LLGCUDA_API int lib_main(lua_State* L)
{
	return 0;
}




