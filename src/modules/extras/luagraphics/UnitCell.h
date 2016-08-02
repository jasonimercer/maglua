#ifndef UNITCELL_H
#define UNITCELL_H

#include "Atom.h"
#include <vector>
#include "Matrix.h"

using namespace std;

//atoms are stored in global coordinates
class UnitCell
{
public:
	UnitCell();
// 	UnitCell(const Vector& a, const Vector& b, const Vector& c);
	~UnitCell();
	
	void anglesLengthsToBasisVectors(
		double alpha, double beta, double gamma,
		double a, double b, double c);
	
	void addAtomGlobalCoorinates(Atom* a);
	void addAtomReducedCoorinates(Atom* a);

	bool applyOperator(const char* xyz);
	bool applyOperator(lua_State* L, int func_idx);

	Vector reducedToGlobal(const Vector& v) const;
	Vector globalToReduced(const Vector& v) const;
	
	void translate(int rx, int ry, int rz); //translate to neighbouring ucs
	
	vector<Atom*> atoms;
	
	Vector A() const;
	Vector B() const;
	Vector C() const;
	
	void setA(const Vector& v);
	void setB(const Vector& v);
	void setC(const Vector& v);
	
	int refcount;  //for lua
	
	Matrix r2g;
	Matrix g2r;

private:
	Vector _A;
	Vector _B;
	Vector _C;
	
	Vector iA;
	Vector iB;
	Vector iC;
};


UnitCell* lua_tounitcell(lua_State* L, int idx);
int  lua_isunitcell(lua_State* L, int idx);
void lua_pushunitcell(lua_State* L, UnitCell* u);
void lua_registerunitcell(lua_State* L);



#endif
