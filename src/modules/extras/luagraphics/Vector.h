#ifndef VECTOR_H
#define VECTOR_H

// #include <gsl/gsl_matrix.h>

#include <iostream>
using namespace std;

extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}
#include "luabaseobject.h"

#include "libLuaGraphics.h"
class LUAGRAPHICS_API Vector : public LuaBaseObject
{
public:
	Vector(double x=0, double y=0, double z=0);
	Vector(const double* r3);
	//Vector(const gsl_vector* v);
	Vector(const Vector& other);
	Vector(const Vector* other);
	~Vector();

	LINEAGE1("Vector")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	
	void  setComponent(unsigned int c, double value);
	void  setX(double v);
	void  setY(double v);
	void  setZ(double v);
	void  set(double x, double y, double z);
	void  set(const double* v3);

	double component(unsigned int c) const;
	double x() const;
	double y() const;
	double z() const;
	
	void zero();

	void swap(Vector& v2);
	
	double length() const;
	double lengthSquared() const;
	
	double dot(const Vector& rhs) const;
	Vector cross(const Vector& rhs) const;
	void normalize(double unity=1.0);
	Vector normalized(double unity=1.0) const;
	void clamp(double maxlength);
	
	static double radiansBetween(const Vector& a, const Vector& b); 
	
	const double* vec() const;

	static Vector min(const Vector& a, const Vector& b);
	static Vector max(const Vector& a, const Vector& b);
	
	static Vector crossProduct(const Vector& a, const Vector& b);
	static double    dotProduct(const Vector& a, const Vector& b);
	
	Vector& operator =(const Vector& rhs);
	Vector& operator+=(const Vector &rhs);
	Vector& operator-=(const Vector &rhs);
	Vector& operator*=(const double value);
	Vector& operator/=(const double value);
	
	const Vector operator+(const Vector &other) const;
	const Vector operator-(const Vector &other) const;
	const Vector operator-() const;
	const Vector operator*(const double value) const;
	const Vector operator/(const double value) const;
	
	bool operator==(const Vector &other) const;
	bool operator!=(const Vector &other) const;

	void projectOnto(const Vector& v);
	void projectOntoPlane(const Vector& n);
	Vector projectedOntoPlane(const Vector& n) const;
	void rotateAbout(const Vector& vec, double theta);
	
	void randomize(double scale=0.2);
	
	void bmRandom(double scale=1.0); //box-muller random normal coordinates

	double distanceToPlane(const Vector& n, const Vector& x);
private:
	double _xyz[4]; //4 for quick component checking
};

ostream& operator<<(ostream& out, const Vector& v);
const Vector operator*(const double m, const Vector& v);


int lua_makevector(lua_State* L, int idx, Vector& v);
int lua_makevector(lua_State* L, int idx, Vector* v);

#endif
