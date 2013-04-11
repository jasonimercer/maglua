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
#include "spinsystem.h"
#include "spinoperation.h"

LLG::LLG(int encode_type)
: LuaBaseObject(encode_type), disableRenormalization(false)
{
	thermalOnlyFirstTerm = true;

}

int LLG::luaInit(lua_State* L)
{
	return 0;
}

LLG::~LLG()
{
	
}

void LLG::encode(buffer* b)
{
	char version = 0;
	encodeChar(version, b);
	encodeInteger(thermalOnlyFirstTerm, b);
	encodeInteger(disableRenormalization, b);

}

int  LLG::decode(buffer* b)
{
	char version = decodeChar(b);
	if(version == 0)
	{
		thermalOnlyFirstTerm = decodeInteger(b);
		disableRenormalization = decodeInteger(b);
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}
		
	return 0;
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
	
	vector<SpinSystem*>* ss[3]; //from, dmdt, to
	vector<SpinSystem*> v1;

	vector<SpinSystem*> v2;
	vector<SpinSystem*> v3;

	ss[0] = &v1;

	SpinOperation::getSpinSystemsAtPosition(L, sys1_pos, *ss[0]);
	
// 	ss[0] = luaT_to<SpinSystem>(L, sys1_pos);
	ss[1] = ss[0];
	ss[2] = ss[0];
	
	//get 2nd and 3rd systems if present
	if(luaT_is<SpinSystem>(L, sys2_pos+0) || lua_istable(L, sys2_pos+0))
	{
		ss[1] = &v2;
		SpinOperation::getSpinSystemsAtPosition(L, sys2_pos+0, *ss[1]);
	}

	if(luaT_is<SpinSystem>(L, sys2_pos+1) || lua_istable(L, sys2_pos+1))
	{
		ss[2] = &v3;
		SpinOperation::getSpinSystemsAtPosition(L, sys2_pos+1, *ss[2]);
	}

	if(ss[0]->size() != ss[1]->size() || ss[1]->size() != ss[2]->size())
	{
		return luaL_error(L, "Size mismatch in number of SpinSystems given in each table");
	}

	if(ss[0]->size() == 0)
	{
		return luaL_error(L, "apply requires 1, 2 or 3 SpinSystems or Tables of SpinSystems (extra boolean argument allowed to control timestep, optional 2nd arg can scale timestep)");
	}

	llg->apply(&(ss[0]->at(0)), scale_dmdt, &(ss[1]->at(0)), &(ss[2]->at(0)), advanceTime, ss[0]->size());
	
	return 0;
}

static int l_setthermalOnlyFirstTerm(lua_State* L)
{
 	LUA_PREAMBLE(LLG, llg, 1);
	
	if(lua_isnone(L, 2))
	{
		//default true
		llg->thermalOnlyFirstTerm = true;
	}
	else
	{
		llg->thermalOnlyFirstTerm = lua_toboolean(L, 2);
	}
	return 0;
}
static int l_getthermalOnlyFirstTerm(lua_State* L)
{
 	LUA_PREAMBLE(LLG, llg, 1);
	lua_pushboolean(L, llg->thermalOnlyFirstTerm);
	return 1;
}

static int l_getdisablerenormalization(lua_State* L)
{
 	LUA_PREAMBLE(LLG, llg, 1);
	lua_pushboolean(L, llg->disableRenormalization);
	return 1;
}
static int l_setdisablerenormalization(lua_State* L)
{
 	LUA_PREAMBLE(LLG, llg, 1);
	
	if(lua_isnone(L, 2))
	{
		//default true
		llg->disableRenormalization = true;
	}
	else
	{
		llg->disableRenormalization = lua_toboolean(L, 2);
	}
	return 0;
}


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
		lua_pushstring(L, "1 *SpinSystem*, Optional Number, Optional 2 *SpinSystem*s, Optional Boolean: " 
		"Make 1 Euler step from 1st system using 2nd system to compute derivative (defaulting to 1st system), "
		"scaling timestep be optional number (default 1.0) and storing in 3rd system (defaulting to 1st system)."
		"If last argument is the boolean \"false\", the time will not be incremented. Each *SpinSystem* can "
		"be replaced by a table of SpinSystems to apply the LLG operator to each system.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_setthermalOnlyFirstTerm)
	{
		lua_pushstring(L, "Set internal data for algorithm interpretation.");
		lua_pushstring(L, "1 Optional boolean: If true or none the thermal term in the"
					" effective field will only be applied to the precesional term of the LLG equation. If false then the"
					" thermal term will be used in both the precesional and damping terms. The default value for this variable is true"
					" this default can be changed by setting the base LLG table's `thermalOnlyFirstTerm' key to a boolean value. Example:\n"
					 "<pre>LLG.thermalOnlyFirstTerm = false\nllg = LLG.Cartesian.new(ss)\nprint(llg:thermalOnlyFirstTerm())</pre>\nOutput:\n<pre>false</pre>");
		lua_pushstring(L, "");
		return 3;		
	}
		
	if(func == l_getthermalOnlyFirstTerm)
	{
		lua_pushstring(L, "Get internal data for algorithm interpretation.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 boolean: If true then the thermal term is excluded from the damping term of the LLG equation, if false"
		" then the thermal term is used in both the precesional and damping terms.");
		return 3;		
	}
	
	if(func == l_getdisablerenormalization)
	{
		lua_pushstring(L, "Get internal data for algorithm interpretation.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 boolean: If true then the x,y,z coordinates of each spin are not renormalized to the original length of the spin.");
		return 3;		
	}
	if(func == l_setdisablerenormalization)
	{
		lua_pushstring(L, "Set internal data for algorithm interpretation.");
		lua_pushstring(L, "1 boolean (or default true): If true then the x,y,z coordinates of each spin are not renormalized to the original length of the spin.");
		lua_pushstring(L, "");
		return 3;		
	}
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
		{"setThermalOnlyFirstTerm", l_setthermalOnlyFirstTerm},
		{"thermalOnlyFirstTerm", l_getthermalOnlyFirstTerm},
		{"setDisableRenormalization", l_setdisablerenormalization},
		{"disableRenormalization", l_getdisablerenormalization},
		
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}



#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef LLGCUDA_EXPORTS
  #define LLGCUDA_API __declspec(dllexport)
 #else
  #define LLGCUDA_API __declspec(dllimport)
 #endif
#else
 #define LLGCUDA_API 
#endif

#include "info.h"
extern "C"
{
LLGCUDA_API int lib_register(lua_State* L);
LLGCUDA_API int lib_version(lua_State* L);
LLGCUDA_API const char* lib_name(lua_State* L);
LLGCUDA_API int lib_main(lua_State* L);
}

#include "llgcartesian.h"
#include "llgquat.h"

int lib_register(lua_State* L)
{
	luaT_register<LLG>(L);
	luaT_register<LLGCartesian>(L);
	luaT_register<LLGQuaternion>(L);
//	luaT_register<LLGAlign>(L);
//	luaT_register<LLGFake>(L);	
	return 0;
}

int lib_version(lua_State* L)
{
	return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "LLG-Cuda";
#else
	return "LLG-Cuda-Debug";
#endif
}

int lib_main(lua_State* L)
{
	return 0;
}


