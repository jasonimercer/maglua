#ifndef MATRIX_H
#define MATRIX_H

// #include <gsl/gsl_matrix.h>

#include <iostream>
using namespace std;

extern "C" {
		#include <lua.h>
		#include <lualib.h>
		#include <lauxlib.h>
}
#include "luabaseobject.h"

class Vector;

#include "libLuaGraphics.h"
class LUAGRAPHICS_API Matrix : public LuaBaseObject
{
public:
	Matrix();
	Matrix(const double* r16);
	Matrix(const Matrix& other);
// 	Matrix(const gsl_matrix* m);
	~Matrix();
	
	LINEAGE1("Matrix")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);

	void  setComponent(unsigned int r, unsigned int c, double value);
	void  setComponent(unsigned int i, double value);
	
	double component(unsigned int r, unsigned int c) const;
	double component(unsigned int i) const;
	
	void makeTranslation(double dx, double dy, double dz);
	void makeTranslation(const Vector& v);
	void makeRotationR(double radians, double x, double y, double z);
	void makeRotationD(double degrees, double x, double y, double z);
	void makeRotationR(double radians, const Vector& v);
	void makeRotationD(double degrees, const Vector& v);
	void makeScale(double dx, double dy, double dz);
	void makeScale(const Vector& v);
	
	
	void makeZero();
	void makeIdentity();
	void invert();

	void swap(Matrix& v2);
	
	double det() const;
	
	Matrix multiply(const Matrix& rhs) const;
	
	const double* vec() const;

	Matrix& operator =(const Matrix& rhs);
	Matrix& operator+=(const Matrix &rhs);
	Matrix& operator-=(const Matrix &rhs);
	Matrix& operator*=(const Matrix &rhs);
	Matrix& operator*=(const double value);
// 	Matrix& operator/=(const double value);
	
	const Matrix operator+(const Matrix &other) const;
	const Matrix operator-(const Matrix &other) const;
	const Matrix operator-() const;
	const Matrix operator*(const double value) const;
	const Matrix operator*(const Matrix &rhs) const;
// 	const Matrix operator/(const double value) const;
	
	bool operator==(const Matrix &other) const;
	bool operator!=(const Matrix &other) const;
private:
	double _m16[16];
};

ostream& operator<<(ostream& out, const Matrix& m);
const Matrix operator*(const double a, const Matrix& m);


const Vector operator*(const Matrix& lhs, const Vector& rhs);


int lua_makematrix(lua_State* L, int idx, Matrix& m);


#endif
