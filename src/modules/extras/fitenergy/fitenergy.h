#ifndef _FE_DEF
#define _FE_DEF

#include "luabaseobject.h"
#include <vec_cs.h>
#include <vector>
using namespace std;


class FitEnergy : public LuaBaseObject
{
public:
    FitEnergy();
    ~FitEnergy();
    
    LINEAGE1("FitEnergy");
    static const luaL_Reg* luaMethods();
    virtual int luaInit(lua_State* L, int base=1);
    static int help(lua_State* L);

    virtual void encode(buffer* b);
    virtual int  decode(buffer* b);

    void addTerm(int a, int b, int i, int j)
    {
        terms.push_back(a);
        terms.push_back(b);
        terms.push_back(i);
        terms.push_back(j);
    }

    double eval(lua_State* L, int base);
    double eval(const vector<VectorCS>& v);
    double eval(int n, const double* v);

    vector<int> terms;    
    vector<double> x;

    int ref_energy_function;
    int ref_data;
};

#endif
