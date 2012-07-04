#ifndef FMMFMMOctTree_H
#define FMMFMMOctTree_H

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

#include "fmm_math.h"
using namespace std;

class MULTIPOLE_API FMMOctTree : public LuaBaseObject
{
public:
	FMMOctTree(const int max_degree=5, dArray*  x=0, dArray*  y=0, dArray*  z=0,
			  dArray* sx=0, dArray* sy=0, dArray* sz=0, FMMOctTree* parent = 0);
	void init(const int max_degree=5, dArray*  x=0, dArray*  y=0, dArray*  z=0,
			  dArray* sx=0, dArray* sy=0, dArray* sz=0, FMMOctTree* parent = 0);
	~FMMOctTree();

	LINEAGE1("FMMOctTree")
    static const luaL_Reg* luaMethods();
    virtual void push(lua_State* L);
    virtual int luaInit(lua_State* L);
    static int help(lua_State* L);

	bool contains(double px, double py, double pz);
	void getStats(double* meanXYZ, double* stddevXYZ);
	void setBounds(double* low, double* high, int childNumber);
	void calcLocalOrigin();
	void split(int until_contains=0);

	void calcInnerTensor(double epsilon=1e-6);
	void calcChildTranslationTensorOperators();

	void fieldAt(double* p3, double* h3);

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
	double localOrigin[3];

	FMMOctTree* c[8]; //children (lowz highz) (lowy highy) (lowx highx)
	FMMOctTree* parent;

	complex<double>* inner;
	int inner_length;

	complex<double>** child_translation_tensor;

	int max_degree;
};

#endif
