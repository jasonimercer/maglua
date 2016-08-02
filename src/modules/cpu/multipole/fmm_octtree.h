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

class FMMOctTreeWorkSpace
{
public:
	FMMOctTreeWorkSpace(int max_degree);
	~FMMOctTreeWorkSpace();
	complex<double>* tensorA;
	complex<double>* tensorB;
	complex<double>* tensorC;
};

class MULTIPOLE_API FMMOctTree : public LuaBaseObject
{
public:
	FMMOctTree(const int max_degree=5,
			   dArray*  x=0, dArray*  y=0, dArray*  z=0,
			   dArray* sx=0, dArray* sy=0, dArray* sz=0,
			   dArray* hx=0, dArray* hy=0, dArray* hz=0,
			   FMMOctTree* parent = 0);
	void init(const int max_degree=5,
			  dArray*  x=0, dArray*  y=0, dArray*  z=0,
			  dArray* sx=0, dArray* sy=0, dArray* sz=0,
			  dArray* hx=0, dArray* hy=0, dArray* hz=0,
			  FMMOctTree* parent = 0);
	~FMMOctTree();

	LINEAGE1("FMMOctTree")
    static const luaL_Reg* luaMethods();
    virtual int luaInit(lua_State* L);
    static int help(lua_State* L);

    bool contains(double px, double py, double pz);
    bool contains(double* p3);
    void getStats(double* meanXYZ, double* stddevXYZ);
	void setBounds(double* low, double* high, int childNumber);
	void calcLocalOrigin();
	void split(int times);

	void apply();

	void calcDipoleFields();


	void calcInnerTensor(double epsilon=1e-6);
	void calcChildTranslationTensorOperators();

	void fieldAt(double* p3, double* h3);
	int totalChildNodes();
    // pointers to all members (positions)
    dArray* x;
    dArray* y;
    dArray* z;

	// pointers to all members (orientations)
	dArray* sx;
	dArray* sy;
	dArray* sz;

	// pointers to all members (fields)
	dArray* hx;
	dArray* hy;
	dArray* hz;

	// members that belong here
	vector<int> members;

	double bounds_low[3];
	double bounds_high[3];
	double bounds_dims[3];
	double localOrigin[3];

	FMMOctTree* c[8]; //children (lowz highz) (lowy highy) (lowx highx)
	FMMOctTree* parent;

	complex<double>* inner;
	int inner_length;

	complex<double>** child_translation_tensor;

	int max_degree;
	int generation;

	int extra_data;

	FMMOctTreeWorkSpace* WS;

	vector<FMMOctTree*> nodes_near;
	vector<FMMOctTree*> nodes_far;
};

#endif
