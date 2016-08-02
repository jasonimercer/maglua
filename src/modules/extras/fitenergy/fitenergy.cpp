#include "fitenergy.h"
#include "fitenergy_luafuncs.h"
#include "info.h"
#include "luamigrate.h"
#include <math.h>


FitEnergy::FitEnergy()
    : LuaBaseObject(hash32(FitEnergy::slineage(0)))
{
    ref_energy_function = LUA_REFNIL;
    ref_data = LUA_REFNIL;
}

FitEnergy::~FitEnergy()
{
    if(ref_energy_function != LUA_REFNIL)
        luaL_unref(L, LUA_REGISTRYINDEX, ref_energy_function);
    ref_energy_function = LUA_REFNIL;

    // deinit();

    if(ref_data != LUA_REFNIL)
        luaL_unref(L, LUA_REGISTRYINDEX, ref_data);
    ref_data = LUA_REFNIL;
}

int FitEnergy::luaInit(lua_State* L, int base)
{
    return LuaBaseObject::luaInit(L, base);
}

void FitEnergy::encode(buffer* b) //encode to data stream
{
    ENCODE_PREAMBLE
}

int  FitEnergy::decode(buffer* b) // decode from data stream
{
    return 0;
}

int FitEnergy::help(lua_State* L)
{
    lua_CFunction func = lua_tocfunction(L, 1);

    return LuaBaseObject::help(L);
}

double FitEnergy::eval(const vector<VectorCS>& v)
{
    vector<VectorCS> s;
    for(int i=0; i<v.size(); i++)
    {
        s.push_back(v[i].convertedToCoordinateSystem(Cartesian).normalizedTo(1));
    }

    const int numTerms = (terms.size())/4;


    if(x.size() != numTerms)
    {
        return 0;
    }

    double sum = 0;
    for(int q=0; q<numTerms; q++)
    {
        const int a = terms[q*4+0];
        const int b = terms[q*4+1];
        const int i = terms[q*4+2];
        const int j = terms[q*4+3];

        double v1 = 1;
        double v2 = 1;


        if(i >= 0 && i < s.size() && a >= 0 && a < 3)
            v1 = s[i].v[a];

        if(j >= 0 && j < s.size() && b >= 0 && b < 3)
            v2 = s[j].v[b];

        const double Aq = v2 * v1;

        sum += Aq * x[q];
    }

    return sum;
}

double FitEnergy::eval(lua_State* L, int base)
{
    vector<VectorCS> vvcs;
    if(!lua_istable(L, base))
        luaL_error(L, "table expected at base");

    for(int i=1; !lua_isnil(L, -1); i++)
    {
        lua_pushinteger(L, i);
        lua_gettable(L, base);
        if(!lua_isnil(L, -1))
        {
            vvcs.push_back( lua_toVectorCS(L, lua_gettop(L) ) );
            lua_pop(L, 1);
        }
    }
    lua_pushnumber(L, eval(vvcs));
    return 1;
}

double FitEnergy::eval(int n, const double* v)
{
    vector<VectorCS> vvcs;
    for(int i=0; i<n; i++)
    {
        vvcs.push_back(VectorCS(&(v[i*3])));
    }
    return eval(vvcs);
}



static int l_setdata(lua_State* L)
{
    LUA_PREAMBLE(FitEnergy, fe, 1);
    luaL_unref(L, LUA_REGISTRYINDEX, fe->ref_data);
    lua_pushvalue(L, 2);
    fe->ref_data = luaL_ref(L, LUA_REGISTRYINDEX);
    return 0;
}

static int l_getdata(lua_State* L)
{
    LUA_PREAMBLE(FitEnergy, fe, 1);
    lua_rawgeti(L, LUA_REGISTRYINDEX, fe->ref_data);
    return 1;
}

static int l_addterm(lua_State* L)
{
    LUA_PREAMBLE(FitEnergy, fe, 1);
    const int a = lua_tointeger(L, 2) - 1;
    const int b = lua_tointeger(L, 3) - 1;
    const int i = lua_tointeger(L, 4) - 1;
    const int j = lua_tointeger(L, 5) - 1;
    fe->addTerm(a,b,i,j);

    return 0;
}

static int l_clearterms(lua_State* L)
{
    LUA_PREAMBLE(FitEnergy, fe, 1);
    fe->terms.clear();
    return 0;
}

static int l_setx(lua_State* L)
{
    LUA_PREAMBLE(FitEnergy, fe, 1);
    if(!lua_istable(L, 2))
        return luaL_error(L, "Table expected");

    fe->x.clear();
    for(int i=1; !lua_isnil(L, -1); i++)
    {
        lua_pushinteger(L, i);
        lua_gettable(L, 2);
        if(!lua_isnil(L, -1))
        {
            double x = lua_tonumber(L, -1);
            fe->x.push_back(x);
            lua_pop(L, 1);
        }
    }

    return 0;
}

static int l_call(lua_State* L)
{
    LUA_PREAMBLE(FitEnergy, fe, 1);
    return fe->eval(L, 2);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* FitEnergy::luaMethods()
{
    if(m[127].name)return m;
    merge_luaL_Reg(m, LuaBaseObject::luaMethods());

    static const luaL_Reg _m[] =
	{
            {"setInternalData", l_setdata},
            {"getInternalData", l_getdata},
            {"addTerm", l_addterm},
            {"clearTerms", l_clearterms},
            {"setX", l_setx},
            {"__call", l_call},
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
	luaT_register<FitEnergy>(L);

	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");

        luaL_dofile_fitenergy_luafuncs(L);

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
	return "FitEnergy";
#else
	return "FitEnergy-Debug";
#endif
    }

    int lib_main(lua_State* L)
    {
	return 0;
    }
}
