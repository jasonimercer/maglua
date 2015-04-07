// this contains the luabaseobject parts of VectorCS
#include "vec_cs.h"
#include "info.h"

int VectorCS::luaInit(lua_State* L, int base)
{
    if(luaT_is<VectorCS>(L, base))
    {
        (*this) = *luaT_to<VectorCS>(L, base);
    }
    else
    {
        (*this) = lua_toVectorCS(L, base);
    }

    return LuaBaseObject::luaInit(L, base);
}

void VectorCS::encode(buffer* b) //encode to data stream
{
    ENCODE_PREAMBLE;

    for(int i=0; i<3; i++)
        encodeDouble(v[i], b);
    encodeInteger((int)cs, b);
}

int  VectorCS::decode(buffer* b) // decode from data stream
{
    double v0 = decodeDouble(b);
    double v1 = decodeDouble(b);
    double v2 = decodeDouble(b);
    CoordinateSystem cc = (CoordinateSystem)decodeInteger(b);

    set(v0,v1,v2,cc);

    return 0;
}

static int l_scalefactors(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v, 1);

    double sf[3];
    v->scaleFactors(sf);
    for(int i=0; i<3; i++)
        lua_pushnumber(L, sf[i]);
    return 3;
}

static int l_stepsize(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v, 1);
    double epsilon = lua_tonumber(L, 2);

    double sf[3];
    v->stepSize(epsilon, sf);

    for(int i=0; i<3; i++)
        lua_pushnumber(L, sf[i]);
    return 3;
}


// add a random cartesian vector (scaled by m) to the cartesian
// form of the current vector. Rescale to old length, convert back to original CS.
static int l_randomizedirection(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v, 1);
    double m = lua_tonumber(L, 2);
    v->randomizeDirection(m);
    return 0;
}

static int l_project(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v1, 1);
    LUA_PREAMBLE(VectorCS, v2, 2);
    v1->project(*v2);
    return 0;
}

static int l_reject(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v1, 1);
    LUA_PREAMBLE(VectorCS, v2, 2);
    v1->reject(*v2);
    return 0;
}








static int l_normalizeto(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v, 1);

    if(lua_isnumber(L, 2))
        v->setMagnitude(lua_tonumber(L, 2));
    else
        v->setMagnitude(1);

    return 0;
}

static int l_tolist(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v, 1);

    for(int i=0; i<3; i++)
    {
        lua_pushnumber(L, v->v[i]);
    }
    lua_pushstring(L, nameOfCoordinateSystem(v->cs));
    return 4;
}

static int l_fromlist(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v, 1);

    double v0 = lua_tonumber(L, 2);
    double v1 = lua_tonumber(L, 3);
    double v2 = lua_tonumber(L, 4);

    if(lua_gettop(L) >= 5)
    {
        CoordinateSystem cs = coordinateSystemByName(lua_tostring(L, 5));
        v->set(v0,v1,v2,cs);
    }
    else
    {
        v->set(v0,v1,v2,v->cs);
    }
    return 0;
}

static int l_convertcs(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v, 1);

    CoordinateSystem cs = coordinateSystemByName(lua_tostring(L, 2));
    if(cs == Undefined)
        return luaL_error(L, "Unknown coordinate system: %s", lua_tostring(L, 2));
    v->convertToCoordinateSystem(cs);
    return 0;
}

static int l_mag(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v, 1);
    lua_pushnumber(L, v->magnitude());
    return 1;
}

static int l_zeroradialcomponent(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v, 1);
        v->zeroRadialComponent();
    return 0;
}


static int fl_anglebetween(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v1, 1);
    LUA_PREAMBLE(VectorCS, v2, 2);

    lua_pushnumber(L, VectorCS::angleBetween(*v1,*v2));
    return 1;
}
static int fl_arclength(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v1, 1);
    LUA_PREAMBLE(VectorCS, v2, 2);

    lua_pushnumber(L, VectorCS::arcLength(*v1,*v2));
    return 1;
}
static int fl_cross(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v1, 1);
    LUA_PREAMBLE(VectorCS, v2, 2);

    luaT_push<VectorCS>(L, new VectorCS( VectorCS::cross(*v1, *v2) ));
    return 1;
}
static int fl_axpy(lua_State* L)
{
    double a = lua_tonumber(L, 1);
    LUA_PREAMBLE(VectorCS, v1, 2);
    LUA_PREAMBLE(VectorCS, v2, 3);

    luaT_push<VectorCS>(L, new VectorCS( VectorCS::axpy(a, *v1, *v2)));
    return 1;
}

static int l_rotateaboutby(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v1, 1);
    LUA_PREAMBLE(VectorCS, v2, 2);
    double a = lua_tonumber(L, 3);

    v1->rotateAboutBy(*v2, a);
    return 0;
}

static int l_angleBetween(lua_State* L)
{
    LUA_PREAMBLE(VectorCS, v1, 1);
    LUA_PREAMBLE(VectorCS, v2, 2);
    double a = VectorCS::angleBetween(*v1, *v2);

    lua_pushnumber(L, a);
    return 1;
}



static const struct {
    const char *name;
    lua_CFunction func;
    const char *desc;
    const char *input;
    const char *output;
} staticFunctionData[] = {
    {   "angleBetween", fl_anglebetween,
        "Determine the number of radians between the two given vectors.",
        "2 Vectors: Vectors bounding angle.",
        "1 Number: Radians between vectors."
    },
    {   "arcLength", fl_arclength,
        "Determine the length of the arc between the two given vectors.",
        "2 Vectors: Vectors bounding arc.",
        "1 Number: Arc length between vectors."
    },
    {   "cross", fl_cross,
        "Cross two vectors",
        "2 Vectors: Vectors to cross",
        "1 Vector: cross product of vectors"
    },
    {   "axpy", fl_axpy,
        "Add two vectors scaling the first by a value",
        "1 Number, 2 Vectors: Value to scale first vector and two vectors to sum",
        "1 Vector: result of summation."
    },
    {0,0,0,0,0}
};

static const struct {
    const char *name;
    lua_CFunction func;
    const char *desc;
    const char *input;
    const char *output;
} staticMethodData[] = {
    {   "rotateAboutBy", l_rotateaboutby,
        "Rotate the calling vector about another vector by the specified number of radians",
        "1 Vector, 1 Number: Axis to rotate about and rotation amount",
        ""
    },
    {   "scaleFactors", l_scalefactors,
        "Get the scale factors used in differentiation for this vector at it's orientation and magnitude",
        "",
        "3 Numbers: Scale factors"
    },
    {   "angleBetween", l_angleBetween,
        "Get the angle (in radians) between the calling VectorCS and the given VectorCS"
        "1 VectorCS: other vector",
        "1 Number: Angle between vectors"
    },
    {   "stepSize", l_stepsize,
        "Get the step size in each direction taking into account the scaleFactor",
        "1 Number: Base step size",
        "3 Numbers: Scaled step sizes"
    },
    {
        "randomizeDirection", l_randomizedirection,
        "Add a random cartesian vector (scaled by input) to the Cartesian "
        "form of the current vector. Rescale to old length, convert back to original coordinate system.",
        "1 Number: scale",
        ""
    },
    {   "project", l_project,
        "Project the calling vector onto the argument vector",
        "1 Vector: Ray to project the vector onto.",
        ""
    },
    {   "reject", l_reject,
        "Reject the calling vector from the argument vector",
        "1 Vector: Ray to reject the vector from.",
        ""
    },
    {   "magnitude", l_mag, 
        "Get the magnitude of the vector",
        "",
        "1 Number: magnitude"
    },
    {   "setMagnitude", l_normalizeto,
        "Scale vector to given length or 1.", 
        "1 Optional Number: New length(default 1)", 
        ""
    },
    {   "toList", l_tolist, 
        "Get Vector as a list of Lua types",
        "",
        "3 Numbers and 1 String: Coordinates and Coordinate System Name"
    },
    {   "fromList", l_fromlist, 
        "Set Vector from a list of Lua types",
        "3 Numbers and 1 Optional String: Coordinates and Coordinate System Name (default current)",
        ""
    },
    {   "convertTo", l_convertcs,
        "Convert vector to another coordinate system",
        "1 String: New coordinate system",
        ""
    },
    { 0, 0, 0, 0, 0 }
};


int VectorCS::help(lua_State* L)
{
    lua_CFunction func = lua_tocfunction(L, 1);

    for(int i=0; staticMethodData[i].name; i++)
    {
        if(func == staticMethodData[i].func)
        {
            lua_pushstring(L, staticMethodData[i].desc);
            lua_pushstring(L, staticMethodData[i].input);
            lua_pushstring(L, staticMethodData[i].output);
            return 3;
        }
    }

    for(int i=0; staticFunctionData[i].name; i++)
    {
        if(func == staticFunctionData[i].func)
        {
            lua_pushstring(L, staticMethodData[i].desc);
            lua_pushstring(L, staticMethodData[i].input);
            lua_pushstring(L, staticMethodData[i].output);
            return 3;
        }
    }

    return LuaBaseObject::help(L);
}





static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* VectorCS::luaMethods()
{
    if(m[127].name)return m;
    merge_luaL_Reg(m, LuaBaseObject::luaMethods());

    for(int i=0; staticMethodData[i].name; i++)
        merge_luaL_pair(m, staticMethodData[i].name, staticMethodData[i].func);

    m[127].name = (char*)1;
    return m;
}



#include "vec_cs_luafuncs.h"
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
        luaT_register<VectorCS>(L);

        lua_getglobal(L, VectorCS::slineage(0));
        for(int i=0; staticFunctionData[i].name; i++)
        {
            lua_pushstring(L, staticFunctionData[i].name);
            lua_pushcfunction(L, staticFunctionData[i].func);
            lua_settable(L, -3);
        }
        lua_pop(L, 1);


        lua_pushcfunction(L, l_getmetatable);
        lua_setglobal(L, "maglua_getmetatable");

        luaL_dofile_vec_cs_luafuncs(L);

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
        return "VectorCS";
#else
        return "VectorCS-Debug";
#endif
    }

    int lib_main(lua_State* L)
    {
        return 0;
    }
}
