#ifndef _PF_DEF
#define _PF_DEF

#include "luabaseobject.h"
#include "vec_cs.h"
#include <vector>

#include "../../cpu/core/spinsystem.h"
#include "array.h"

class PathFinder : public LuaBaseObject
{
public:
    PathFinder();
    ~PathFinder();
    
    LINEAGE1("PathFinder");
    static const luaL_Reg* luaMethods();
    virtual int luaInit(lua_State* L, int base=1);
    static int help(lua_State* L);

    virtual void encode(buffer* b);
    virtual int  decode(buffer* b);
    
    double energyOfCustomPoint(const vector<VectorCS>& v1);
    int numberOfSites();

    int setEnergyFunction(lua_State* L, int idx);


    int ref_energy_function;
};

#endif
