#ifndef BSPTREE_H
#define BSPTREE_H

#include "array.h"
#include <vector>
using namespace std;

class BSPTree
{
public:
	BSPTree(dArray* x=0, dArray* y=0, dArray* z=0, dArray* weight=0, BSPTree* parent = 0);
	~BSPTree();

	double div;
	void getStats(double* meanXYZ, double* stddevXYZ);
	
	void split(int until_contains=0);
	
	// pointers to all members
	dArray* x;
	dArray* y;
	dArray* z;
	dArray* weight;
	
	// members that belong here
	vector<int> members;

	int split_dir; //0 = x, 1 = y, 2 = z
	double split_value;
	
	BSPTree* c[2]; //children
	BSPTree* parent;
};

#endif
