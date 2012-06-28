#ifndef OCTTREE_H
#define OCTTREE_H

#include "array.h"
#include <vector>
#include "luabaseobject.h"

#ifdef WIN32
 #ifdef MULTIPOLE_EXPORTS
  #define MULTIPOLE_API __declspec(dllexport)
 #else
  #define MULTIPOLE_API __declspec(dllimport)
 #endif
#else
 #define MULTIPOLE_API
#endif

class MULTIPOLE_API OctTree : public LuaBaseObject
{
public:
    OctTree(dArray*  x=0, dArray*  y=0, dArray*  z=0,
            dArray* sx=0, dArray* sy=0, dArray* sz=0, OctTree* parent = 0);
    void init(dArray*  x=0, dArray*  y=0, dArray*  z=0,
              dArray* sx=0, dArray* sy=0, dArray* sz=0, OctTree* parent = 0);
    ~OctTree();


    LINEAGE1("OctTree")
    static const luaL_Reg* luaMethods();
    virtual void push(lua_State* L);
    virtual int luaInit(lua_State* L);
    static int help(lua_State* L);

	bool contains(double px, double py, double pz);
	void getStats(double* meanXYZ, double* stddevXYZ);
	
	void setBounds(double* low, double* high, int childNumber);
	
	void split(int until_contains=0);
	
    // pointers to all members (positions)
    dArray* x;
    dArray* y;
    dArray* z;

    // pointers to all members (orientations)
    dArray* sx;
    dArray* sy;
    dArray* sz;

	// members that belong here
	vector<int> members;

	double bounds_low[3];
	double bounds_high[3];
	
	OctTree* c[8]; //children (lowz highz) (lowy highy) (lowx highx)
	OctTree* parent;
};

#endif
