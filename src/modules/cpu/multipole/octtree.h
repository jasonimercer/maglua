#ifndef OCTTREE_H
#define OCTTREE_H

#include "array.h"
#include <vector>
using namespace std;

class OctTree
{
public:
	OctTree(dArray* x=0, dArray* y=0, dArray* z=0, OctTree* parent = 0);
	~OctTree();

	bool contains(double px, double py, double pz);
	void getStats(double* meanXYZ, double* stddevXYZ);
	
	void setBounds(double* low, double* high, int childNumber);
	
	void split(int until_contains=0);
	
	// pointers to all members
	dArray* x;
	dArray* y;
	dArray* z;
	
	// members that belong here
	vector<int> members;

	double bounds_low[3];
	double bounds_high[3];
	
	OctTree* c[8]; //children (lowz highz) (lowy highy) (lowx highx)
	OctTree* parent;
};

#endif
