// working with 
// Adaptation and Performance of the Cartesian
// Coordinates Fast Multipole Method for Nanomagnetic
// Simulations
// Wen Zhang, Stephan Haas


#ifndef CELL_H
#define CELL_H

#include "array.h"
#include "multipole.h"
#include <vector>
using namespace std;

class Cell
{
public:
	Cell(dArray* x=0, dArray* y=0, dArray* z=0, Cell* parent = 0);
	~Cell();

	bool contains(double px, double py, double pz);
	void getStats(double* meanXYZ, double* stddevXYZ);
	
	void setBounds(double* low, double* high, int split_dir, int childNumber);
	
	void split(int split_dir, int until_contains=0);
	
	bool near(const Cell* other) const;
	double openingAngle(const Cell* other) const;
	
	double radius() const;
	double centerCenterDistanceTo(const Cell* other) const;

	void getCenter(double* c3) const;
	void updateMoment();
	
	// pointers to all members
	dArray* x;
	dArray* y;
	dArray* z;
	
	// members that belong here
	vector<int> members;

	double bounds_low[3];
	double bounds_high[3];
	
	vector<Cell*> partners;
	vector<Cell*> nearPartners;
	
	int removeFromPartnerList(Cell* c);
	void createPartners();

	Cell* c[2];
	Cell* parent;
	
	//need smooth field: set of taylor coefficients
	MultipoleCartesian* moment;
	MultipoleCartesian* smoothField;
};



#endif
