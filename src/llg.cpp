#include "llg.h"
#include "llgcartesian.h"
#include "llgquat.h"
#include "llgfake.h"
#include <string.h>

LLG::LLG(const char* llgtype, int etype)
	: Encodable(etype), alpha(0.1), dt(0.01), gamma(1.0), type(llgtype), refcount(0)
{

}

void LLG::encode(buffer* b) const
{
	encodeDouble(alpha, b);
	encodeDouble(   dt, b);
	encodeDouble(gamma, b);

	int len = type.length()+1;
	encodeInteger( len, b);
	encodeBuffer(type.c_str(), len, b);
}

int  LLG::decode(buffer* b)
{
	alpha = decodeDouble(b);
	   dt = decodeDouble(b);
	gamma = decodeDouble(b);
	
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
		llg = new LLGFake;
	}
	if(!llg)
	  return luaL_error(L, "Unknown LLG type `%s'", lua_tostring(L, 1));

	lua_pushLLG(L, llg);
	
	return 1;
}




int l_llg_apply(lua_State* L)
{
	if(lua_gettop(L) == 2)
	{
		LLG* llg = checkLLG(L, 1);
		SpinSystem* ss  = checkSpinSystem(L, 2);
		
		if(!llg)
			return 0;
	
		if(!ss)
			return 0;
	
		llg->apply(ss, ss, ss);
		return 0;
	}
	if(lua_gettop(L) == 4)
	{
		LLG* llg = checkLLG(L, 1);
		SpinSystem* spinfrom  = checkSpinSystem(L, 2);
		SpinSystem* fieldfrom = checkSpinSystem(L, 3);
		SpinSystem* spinto    = checkSpinSystem(L, 4);
		
		if(!llg)
			return 0;
	
		if(!spinfrom || !fieldfrom || !spinto)
			return 0;
	
		llg->apply(spinfrom, fieldfrom, spinto);
	}
	luaL_error(L, "apply requires 1 or 3 spin systems");

	return 0;
}


int l_llg_settimestep(lua_State* L)
{
	LLG* llg = checkLLG(L, 1);
	if(!llg) return 0;
	llg->dt = lua_tonumber(L, 2);
	return 0;
}
int l_llg_gettimestep(lua_State* L)
{
	LLG* llg = checkLLG(L, 1);
	if(!llg) return 0;
	lua_pushnumber(L, llg->dt);
	return 1;
}

int l_llg_setalpha(lua_State* L)
{
	LLG* llg = checkLLG(L, 1);
	if(!llg) return 0;
	llg->alpha = lua_tonumber(L, 2);
	return 0;
}
int l_llg_getalpha(lua_State* L)
{
	LLG* llg = checkLLG(L, 1);
	if(!llg) return 0;
	lua_pushnumber(L, llg->alpha);
	return 1;
}

int l_llg_setgamma(lua_State* L)
{
	LLG* llg = checkLLG(L, 1);
	if(!llg) return 0;
	llg->gamma = lua_tonumber(L, 2);
	return 0;
}
int l_llg_getgamma(lua_State* L)
{
	LLG* llg = checkLLG(L, 1);
	if(!llg) return 0;
	lua_pushnumber(L, llg->gamma);
	return 1;
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

void registerLLG(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_llg_gc},
		{"__tostring",   l_llg_tostring},
		{"apply",        l_llg_apply},
		{"setAlpha",     l_llg_setalpha},
		{"alpha",        l_llg_getalpha},
		{"setTimeStep",  l_llg_settimestep},
		{"timeStep",     l_llg_gettimestep},
		{"setGamma",     l_llg_setgamma},
		{"gamma",        l_llg_getgamma},
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
		{NULL, NULL}
	};
		
	luaL_register(L, "LLG", functions);
	lua_pop(L,1);	
}


