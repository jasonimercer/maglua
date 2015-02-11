#include "pf.h"
#include "pf_luafuncs.h"
#include "info.h"
#include "luamigrate.h"
#include <math.h>

#include <algorithm>
#include <numeric>

#define CARTESIAN_X 0
#define CARTESIAN_Y 1
#define CARTESIAN_Z 2

#define SPHERICAL_R 0
#define SPHERICAL_PHI 1
#define SPHERICAL_THETA 2

#define CANONICAL_R 0
#define CANONICAL_PHI 1
#define CANONICAL_P 2



PathFinder::PathFinder()
    : LuaBaseObject(hash32(PathFinder::slineage(0)))
{
    ref_energy_function = LUA_REFNIL;
}

PathFinder::~PathFinder()
{
    if(ref_energy_function != LUA_REFNIL)
        luaL_unref(L, LUA_REGISTRYINDEX, ref_energy_function);

    ref_energy_function = LUA_REFNIL;

    // deinit();
	
}


int PathFinder::setEnergyFunction(lua_State* L, int idx)
{
    if(ref_energy_function != LUA_REFNIL)
        luaL_unref(L, LUA_REGISTRYINDEX, ref_energy_function);

    ref_energy_function = luaL_ref(L, LUA_REGISTRYINDEX);
    return 0;
}

int PathFinder::luaInit(lua_State* L, int base)
{
    return LuaBaseObject::luaInit(L, base);
}

void PathFinder::encode(buffer* b) //encode to data stream
{
    ENCODE_PREAMBLE
}

int  PathFinder::decode(buffer* b) // decode from data stream
{
    return 0;
}

double PathFinder::energyOfCustomPoint(const vector<VectorCS>& v1)
{
    lua_rawgeti(L, LUA_REGISTRYINDEX, ref_energy_function);

    for(int i=0; i<v1.size(); i++)
    {
        VectorCS x = v1[i].convertedToCoordinateSystem(Cartesian);
        lua_pushVectorCS(L, x, VCSF_ASTABLE);
    }

    lua_call(L, v1.size(), 1);

    double e = lua_tonumber(L, -1);
    lua_pop(L, 1);
    
    return e;
}

int PathFinder::numberOfSites()
{
    return 2;
}

#include "pf_bestpath.h"
static int _l_bestpath(lua_State* L)
{
    return l_pf_bestpath(L);
}

static int l_set_ef(lua_State* L)
{
    LUA_PREAMBLE(PathFinder, pf, 1);
    pf->setEnergyFunction(L, 2);
    return 0;
}


int PathFinder::help(lua_State* L)
{
    lua_CFunction func = lua_tocfunction(L, 1);

/*
    if(func == l_setdata)
    {
	lua_pushstring(L, "Set internal data. This method is used in the support lua scripts and should not be used.");
	lua_pushstring(L, "1 Value: the new internal data");
	lua_pushstring(L, "");
	return 3;
    }
*/

    return LuaBaseObject::help(L);
}





static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* PathFinder::luaMethods()
{
    if(m[127].name)return m;
    merge_luaL_Reg(m, LuaBaseObject::luaMethods());

    static const luaL_Reg _m[] =
	{
            {"setEnergyFunction", l_set_ef},
            {"_findBestPath",  _l_bestpath},
	    {NULL, NULL}
	};
    merge_luaL_Reg(m, _m);
    m[127].name = (char*)1;
    return m;
}






extern "C"
{
    static int l_getmetatable(lua_State* L)
    {
	if(!lua_isstring(L, 1))
	    return luaL_error(L, "First argument must be a metatable name");
	luaL_getmetatable(L, lua_tostring(L, 1));
	return 1;
    }

    int lib_register(lua_State* L)
    {
	luaT_register<PathFinder>(L);

	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");

        luaL_dofile_pf_luafuncs(L);

	lua_pushnil(L);
	lua_setglobal(L, "maglua_getmetatable");


	return 0;
    }

    int lib_version(lua_State* L)
    {
	return __revi;
    }

    const char* lib_name(lua_State* L)
    {
#if defined NDEBUG || defined __OPTIMIZE__
	return "Pathfinder";
#else
	return "PathFinder-Debug";
#endif
    }

    int lib_main(lua_State* L)
    {
	return 0;
    }
}
