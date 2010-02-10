#include "spinoperationappliedfield.h"
#include "spinsystem.h"

#include <stdlib.h>

AppliedField::AppliedField(int nx, int ny, int nz)
	: SpinOperation("AppliedField", APPLIEDFIELD_SLOT, nx, ny, nz)
{
	B[0] = 0;
	B[1] = 0;
	B[2] = 0;
}

AppliedField::~AppliedField()
{
}

bool AppliedField::apply(SpinSystem* ss)
{
	double* hx = ss->hx[slot];
	double* hy = ss->hy[slot];
	double* hz = ss->hz[slot];

	for(int i=0; i<nxyz; i++)
	{
		hx[i] = B[0];
		hy[i] = B[1];
		hz[i] = B[2];
	}
	return true;
}







AppliedField* checkAppliedField(lua_State* L, int idx)
{
	AppliedField** pp = (AppliedField**)luaL_checkudata(L, idx, "MERCER.appliedfield");
    luaL_argcheck(L, pp != NULL, 1, "`AppliedField' expected");
    return *pp;
}


int l_ap_new(lua_State* L)
{
	if(lua_gettop(L) != 3)
		return luaL_error(L, "AppliedField.new requires nx, ny, nz");

	AppliedField* ap = new AppliedField(
			lua_tointeger(L, 1),
			lua_tointeger(L, 2),
			lua_tointeger(L, 3)
	);
	ap->refcount++;
	
	AppliedField** pp = (AppliedField**)lua_newuserdata(L, sizeof(AppliedField**));
	
	*pp = ap;
	luaL_getmetatable(L, "MERCER.appliedfield");
	lua_setmetatable(L, -2);
	return 1;
}

int l_ap_gc(lua_State* L)
{
	AppliedField* ap = checkAppliedField(L, 1);
	if(!ap) return 0;
	
	ap->refcount--;
	if(ap->refcount == 0)
		delete ap;
	
	return 0;
}

int l_ap_apply(lua_State* L)
{
	AppliedField* ap = checkAppliedField(L, 1);
	SpinSystem* ss = checkSpinSystem(L, 2);
	
	if(!ap->apply(ss))
		return luaL_error(L, ap->errormsg.c_str());
	
	return 0;
}

int l_ap_member(lua_State* L)
{
	AppliedField* ap = checkAppliedField(L, 1);
	if(!ap) return 0;

	int px = lua_tointeger(L, 2) - 1;
	int py = lua_tointeger(L, 3) - 1;
	int pz = lua_tointeger(L, 4) - 1;
	
	if(ap->member(px, py, pz))
		lua_pushboolean(L, 1);
	else
		lua_pushboolean(L, 0);

	return 1;
}

int l_ap_set(lua_State* L)
{
	AppliedField* ap = checkAppliedField(L, 1);
	if(!ap) return 0;

	ap->B[0] = lua_tonumber(L, 2);
	ap->B[1] = lua_tonumber(L, 3);
	ap->B[2] = lua_tonumber(L, 4);

	return 0;
}


void registerAppliedField(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_ap_gc},
		{"apply",        l_ap_apply},
		{"set",          l_ap_set},
		{"member",       l_ap_member},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.appliedfield");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_ap_new},
		{NULL, NULL}
	};
		
	luaL_register(L, "AppliedField", functions);
	lua_pop(L,1);	
}

