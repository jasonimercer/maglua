#include <math.h>
#include "Matrix.h"
#include "Vector.h"
#include <stdlib.h>


Matrix::Matrix(const double* r16)
	: LuaBaseObject(hash32("Matrix"))
{
	for(int i=0; i<16; i++)
	{
		_m16[i] = r16[i];
	}
}


int Matrix::luaInit(lua_State* L)
{
	lua_makematrix(L, 1, *this);
	return 0;
}

void Matrix::push(lua_State* L)
{
	luaT_push<Matrix>(L, this);
}



	
	
// Matrix::Matrix(const gsl_matrix* m)
// {
// 	makeIdentity();
// 	for(int r=0; r<4 && r<m->size1; r++)
// 	{
// 		for(int c=0; c<4 && c<m->size2; c++)
// 		{
// 			setComponent(r, c, gsl_matrix_get(m, r, c));
// 		}
// 	}
// 
// 	refcount = 0;
// }


Matrix::Matrix(const Matrix& other)
	: LuaBaseObject(hash32("Matrix"))
{
	*this = other;
}

Matrix::Matrix()
	: LuaBaseObject(hash32("Matrix"))
{
	makeZero();
}


Matrix::~Matrix()
{
}


void Matrix::swap(Matrix& m2)
{
	double f;
	for(int i=0; i<16; i++)
	{
		f = component(i);
		setComponent(i, m2.component(i));
		m2.setComponent(i, f);
	}
}

#include <stdio.h>
void Matrix::setComponent(unsigned int c, double value)
{
	if(c < 0)
		c = 0;
	if(c > 15)
		c = 15;
	_m16[c] = value;
}

void Matrix::setComponent(unsigned int r, unsigned int c, double value)
{
	setComponent(r*4+c, value);
}


static double randf()
{
	int i = rand();
	double r = (double)i / (double)RAND_MAX;
	
	return r;
}
	
double Matrix::component(unsigned int i) const
{
	if(i<0)
		return component(0);
	if(i > 15)
		return component(15);
	return _m16[i];
}

double Matrix::component(unsigned int r, unsigned int c) const
{
	return component(r*4+c);
}


// 0 1 2
// 3 4 5
// 6 7 8
static double det33(const double* m)
{
	return m[0]*(m[4]*m[8] - m[5]*m[7]) 
		 - m[1]*(m[3]*m[8] - m[5]*m[6])
		 + m[2]*(m[3]*m[7] - m[4]*m[6]);
}

static void subMatrix(const double* m16, double* m9, 
	const int i0, const int i1, const int i2, 
	const int i3, const int i4,	const int i5, 
	const int i6, const int i7, const int i8)
{
	m9[0] = m16[i0];	m9[1] = m16[i1];	m9[2] = m16[i2];
	m9[3] = m16[i3];	m9[4] = m16[i4];	m9[5] = m16[i5];
	m9[6] = m16[i6];	m9[7] = m16[i7];	m9[8] = m16[i8];
}

// 0 1 2 3
// 4 5 6 7
// 8 9 10 11
// 12 13 14 15
// can be made better
double Matrix::det() const
{
	double m9[9];
	double d = 0;
	
	subMatrix(_m16, m9, 5, 6, 7, 9, 10, 11, 13, 14, 15);
	d += _m16[0] * det33(m9);
	
	subMatrix(_m16, m9, 1, 2, 3, 9, 10, 11, 13, 14, 15);
	d += -_m16[4] * det33(m9);
	
	subMatrix(_m16, m9, 1, 2, 3, 5, 6, 7, 13, 14, 15);
	d += _m16[8] * det33(m9);
	
	subMatrix(_m16, m9, 1, 2, 3, 5, 6, 7, 9, 10, 11);
	d += -_m16[12]* det33(m9);

	return d;
}

const double* Matrix::vec() const
{
	return _m16;
}





Matrix& Matrix::operator=(const Matrix& rhs)
{
	for(int i=0; i<16; i++)
		setComponent(i, rhs.component(i));
	return *this;
}


Matrix& Matrix::operator+=(const Matrix &rhs)
{
	for(int i=0; i<16; i++)
		setComponent(i, component(i) + rhs.component(i));
	return *this;
}

Matrix& Matrix::operator-=(const Matrix &rhs)
{
	for(int i=0; i<16; i++)
		setComponent(i, component(i) - rhs.component(i));
	return *this;
}

Matrix& Matrix::operator*=(const double value)
{
	for(int i=0; i<16; i++)
		setComponent(i, value * component(i));
	return *this;
}

Matrix& Matrix::operator*=(const Matrix &rhs)
{
	*this = *this * rhs;

	return *this;
}

// Matrix& Matrix::operator/=(const double value)
// {
// 	double ivalue = 1.0 / value;
// 	for(int i=0; i<3; i++)
// 		setComponent(i, component(i) * ivalue);
// 	return *this;
// }

const Matrix Matrix::operator+(const Matrix &other) const
{
 Matrix result = *this;
 result += other;
 return result;
}

const Matrix Matrix::operator-(const Matrix &other) const
{
	Matrix result = *this;
	result -= other;
	return result;
}

const Matrix Matrix::operator-() const
{
	Matrix result = *this;
	return -1.0*result;
}

const Matrix operator*(const double m, const Matrix& v)
{
	return v*m;
}

const Matrix Matrix::operator*(const double value) const
{
	Matrix result = *this;
	result *= value;
	return result;
}

const Matrix Matrix::operator*(const Matrix &rhs) const
{
	Matrix result;

	for(int r=0; r<4; r++)
		for(int c=0; c<4; c++)
		{
			double s = 0;
			for(int i=0; i<4; i++)
			{
				s += component(r, i) * rhs.component(i, c);
			}
			result.setComponent(r, c, s);
		}
	return result;
}

const Vector operator*(const Matrix& m, const Vector& x)
{
	Vector b;
	double v4[4] = {0,0,0,0};
	
	for(int r=0; r<4; r++)
	{
		for(int c=0; c<3; c++)
		{
			v4[r] += m.component(r, c) * x.component(c);
		}
		v4[r] += m.component(r, 3);
	}
	
	if(v4[3] != 0)
	{
		v4[3] = 1.0 / v4[3];
		for(int i=0; i<3; i++)
		{
			b.setComponent(i, v4[i] * v4[3]);
		}
	}
	
	return b;
}


// const Matrix Matrix::operator/(const double value) const
// {
// 	Matrix result = *this;
// 	result /= value;
// 	return result;
// }

bool Matrix::operator==(const Matrix &other) const
{
	for(int i=0; i<16; i++)
		if(component(i) != other.component(i))
			return false;
	return true;
}

bool Matrix::operator!=(const Matrix &other) const
{
 return !(*this == other);
}


ostream& operator<<(ostream& out, const Matrix& v)
{
	for(int r=0; r<4; r++)
	{
		for(int c=0; c<4; c++)
		{
			out << v.component(r, c);
			if(c < 3)
				out << ", ";
		}
		if(r < 3)
			out << endl;
	}

	return out;
}


void Matrix::makeZero()
{
	for(int i=0; i<16; i++)
	{
		_m16[i] = 0;
	}
}

void Matrix::makeIdentity()
{
	for(int i=0; i<16; i++)
		_m16[i] = 0;

	for(int i=0; i<4; i++)
		_m16[i*4+i] = 1;
}

void Matrix::invert()
{
	double d = det();
	
	if(d == 0)
	{
		makeZero();
		printf("FAILED TO INVERT\n");
		return;
	}


	const double* m = _m16;
	double inv[16];
	
	inv[0] = m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] + m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10];
	inv[4] = -m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] - m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10];
	inv[8] = m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] + m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9];
	inv[12] = -m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] - m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9];
	inv[1] = -m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] - m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10];
	inv[5] = m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] + m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10];
	inv[9] = -m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] - m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9];
	inv[13] = m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] + m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9];
	inv[2] = m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] + m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6];
	inv[6] = -m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] - m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6];
	inv[10] = m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] + m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5];
	inv[14] = -m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] - m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5];
	inv[3] = -m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] - m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6];
	inv[7] = m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] + m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6];
	inv[11] = -m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] - m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5];
	inv[15] = m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] + m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5];

	d = 1.0 / d;

	for(int i=0; i<16; i++)
		_m16[i] = inv[i] * d;
}

void Matrix::makeTranslation(double dx, double dy, double dz)
{
	makeIdentity();
	_m16[ 3] = dx;
	_m16[ 7] = dy;
	_m16[11] = dz;
}

void Matrix::makeTranslation(const Vector& v)
{
	makeTranslation(v.x(), v.y(), v.z());
}

void Matrix::makeRotationR(double radians, double x, double y, double z)
{
	double len = sqrt(x*x+y*y+z*z);
	if(len == 0)
	{
		makeIdentity();
		return;
	}
	
	len = 1.0 / len;
	x *= len;
	y *= len;
	z *= len;

	double c = cos(radians);
	double s = sin(radians);
	
	makeIdentity();
	
// 	x2(1−c)+c  xy(1−c)−zs xz(1−c)+ys 0
// 
// 	yx(1−c)+zs y2(1−c)+c  yz(1−c)−xs 0
// 
// 	xz(1−c)−ys yz(1−c)+xs z2(1−c)+c  0
// 
// 		0          0          0      1

	_m16[ 0]=x*x*(1-c)+c;   _m16[ 1]=x*y*(1-c)-z*s; _m16[ 2]=x*z*(1-c)+y*s; 
	_m16[ 4]=y*x*(1-c)+z*s; _m16[ 5]=y*y*(1-c)+c;   _m16[ 6]=y*z*(1-c)-x*s;
	_m16[ 8]=x*z*(1-c)-y*s; _m16[ 9]=y*z*(1-c)+x*s; _m16[10]=z*z*(1-c)+c;
}

void Matrix::makeRotationD(double degrees, double x, double y, double z)
{
	makeRotationR(degrees * 3.141592653/180.0, x, y, z);
}

void Matrix::makeRotationR(double radians, const Vector& v)
{
	makeRotationR(radians, v.x(), v.y(), v.z());
}

void Matrix::makeRotationD(double degrees, const Vector& v)
{
	makeRotationD(degrees, v.x(), v.y(), v.z());
}

void Matrix::makeScale(double dx, double dy, double dz)
{
	makeIdentity();
	_m16[ 0] = dx;
	_m16[ 5] = dy;
	_m16[10] = dz;
}

void Matrix::makeScale(const Vector& v)
{
	makeScale(v.x(), v.y(), v.z());
}
















int lua_makematrix(lua_State* L, int idx, Matrix& v)
{
	if(luaT_is<Matrix>(L, idx))
	{
		Matrix* b = luaT_to<Matrix>(L, idx);
		if(b)
		{
			v = *b;
		}
		return 1;
	}
	
	if(lua_istable(L, idx))
	{
		
		for(int r=0; r<4; r++)
		{
			lua_pushinteger(L, 1+r);
			lua_gettable(L, idx); //now have the row table on top
			for(int c=0; c<4; c++)
			{
				lua_pushinteger(L, 1+c);
				lua_gettable(L, -2);
				v.setComponent(r, c, lua_tonumber(L, -1));
				lua_pop(L, 1);
			}
			lua_pop(L, 1); //pop row table
		}
		return 1;
	}
	
	for(int i=0; i<16; i++)
	{
		v.setComponent(i, lua_tonumber(L, i+idx));
	}
	return 16;
}


static int l_tostring(lua_State* L)
{
	LUA_PREAMBLE(Matrix, v, 1);
	const double* m = v->vec();
	
	lua_pushfstring(L, 
			"{{%f, %f, %f, %f}, {%f, %f, %f, %f}, {%f, %f, %f, %f}, {%f, %f, %f, %f}}",
			m[ 0], m[ 1], m[ 2], m[ 3],
			m[ 4], m[ 5], m[ 6], m[ 7],
			m[ 8], m[ 9], m[10], m[11],
			m[12], m[13], m[14], m[15]);
	return 1;
}

static int l_add(lua_State* L)
{
	LUA_PREAMBLE(Matrix, a, 1);
	LUA_PREAMBLE(Matrix, b, 2);

	luaT_push<Matrix>(L, new Matrix(*a + *b));
	return 1;
}
static int l_sub(lua_State* L)
{
	LUA_PREAMBLE(Matrix, a, 1);
	LUA_PREAMBLE(Matrix, b, 2);

	luaT_push<Matrix>(L, new Matrix(*a - *b));
	return 1;
}
static int l_unm(lua_State* L)
{
	LUA_PREAMBLE(Matrix, a, 1);
	
	luaT_push<Matrix>(L, new Matrix(-1 * (*a)));
	return 1;
}
static int l_mul(lua_State* L)
{
	if(luaT_is<Matrix>(L, 1) && lua_isnumber(L, 2))
	{
		LUA_PREAMBLE(Matrix, a, 1);
		double  s = lua_tonumber(L, 2);
		luaT_push<Matrix>(L, new Matrix((*a) * (s)));
		return 1;
	}

	if(luaT_is<Matrix>(L, 1) && luaT_is<Matrix>(L, 2))
	{
		LUA_PREAMBLE(Matrix, a, 1);
		LUA_PREAMBLE(Matrix, b, 2);
		luaT_push<Matrix>(L, new Matrix((*a) * (*b)));
		return 1;
	}
	
	if(luaT_is<Matrix>(L, 1) && luaT_is<Vector>(L, 2))
	{
		LUA_PREAMBLE(Matrix, a, 1);
		LUA_PREAMBLE(Vector, x, 2);
		luaT_push<Vector>(L, new Vector((*a) * (*x)));
		return 1;
	}
	
	return luaL_error(L, "don't know how to deal with these args yet (%s, %s)", 
					  lua_typename(L, lua_type(L, 1)), 
					  lua_typename(L, lua_type(L, 2))
 					);
// 	if(!a) return 0;
	
// 	luaT_push<Matrix>(L, new Matrix((*a) * lua_tonumber(L, 2)));
// 	return 1;
}
static int l_eq(lua_State* L)
{
LUA_PREAMBLE(Matrix, a, 1);
LUA_PREAMBLE(Matrix, b, 2);

	lua_pushboolean(L, (*a) == (*b));
	return 1;
}


static int l_set(lua_State* L)
{
LUA_PREAMBLE(Matrix, v, 1);
	
	int r = lua_tointeger(L, 2) - 1;
	int c = lua_tointeger(L, 3) - 1;
	
	if(r < 0 || r >= 4)	return luaL_error(L, "row must between 1 and 4");
	if(c < 0 || c >= 4)	return luaL_error(L, "column must between 1 and 4");
	
	v->setComponent(r, c, lua_tonumber(L, 4));
	
	return 0;
}

static int l_get(lua_State* L)
{
LUA_PREAMBLE(Matrix, v, 1);
	
	int r = lua_tointeger(L, 2) - 1;
	int c = lua_tointeger(L, 3) - 1;
	
	if(r < 0 || r >= 4)	return luaL_error(L, "row must between 1 and 4");
	if(c < 0 || c >= 4)	return luaL_error(L, "column must between 1 and 4");
	
	lua_pushnumber(L, v->component(r, c));
	return 1;
}

static int l_zero(lua_State* L)
{
LUA_PREAMBLE(Matrix, v, 1);

	v->makeZero();
	return 0;
}

static int l_identity(lua_State* L)
{
LUA_PREAMBLE(Matrix, v, 1);

	v->makeIdentity();
	return 0;
}


static int l_invert(lua_State* L)
{
LUA_PREAMBLE(Matrix, v, 1);

	v->invert();
	return 0;
}


static int l_inverse(lua_State* L)
{
LUA_PREAMBLE(Matrix, v, 1);

	Matrix* i = new Matrix(*v);
		
	i->invert();
	luaT_push<Matrix>(L, i);
	return 1;
}



static int l_det(lua_State* L)
{
LUA_PREAMBLE(Matrix, v, 1);
	
	lua_pushnumber(L, v->det());
	
	return 1;
}


static int l_trans(lua_State* L)
{
	LUA_PREAMBLE(Matrix, m, 1);
	Vector v;
	lua_makevector(L, 2, v);
	
	m->makeTranslation(v);
	
	return 0;
}

static int l_scale(lua_State* L)
{
	LUA_PREAMBLE(Matrix, m, 1);

	Vector v;
	lua_makevector(L, 2, v);
	
	m->makeScale(v);
	
	return 0;
}

static int l_rotate(lua_State* L)
{
	LUA_PREAMBLE(Matrix, m, 1);

	double degrees = lua_tonumber(L, 2);
	
	Vector v;
	lua_makevector(L, 3, v);
	
	m->makeRotationD(degrees, v);
	
	return 0;
}

static int l_list(lua_State* L)
{
	LUA_PREAMBLE(Matrix, m, 1);

	lua_newtable(L);
	for(int i=0; i<16; i++)
	{
		lua_pushinteger(L, i+1);
		lua_pushnumber(L, m->component(i));
		lua_settable(L, -3);
	}
	return 1;
}

static int l_getrow(lua_State* L)
{
	LUA_PREAMBLE(Matrix, m, 1);

	int r = lua_tointeger(L, 2) - 1;
	
	lua_newtable(L);
	for(int c=0; c<4; c++)
	{
		lua_pushinteger(L, c+1);
		lua_pushnumber(L, m->component(r, c));
		lua_settable(L, -3);
	}
	return 1;
}


static int l_getcol(lua_State* L)
{
	LUA_PREAMBLE(Matrix, m, 1);

	int c = lua_tointeger(L, 2) - 1;
	
	lua_newtable(L);
	for(int r=0; r<4; r++)
	{
		lua_pushinteger(L, c+1);
		lua_pushnumber(L, m->component(r, c));
		lua_settable(L, -3);
	}
	return 1;
}

static int l_frobnorm(lua_State* L)
{
	LUA_PREAMBLE(Matrix, m, 1);

	double d = 0;
	for(int i=0; i<16; i++)
	{
		d += pow(m->component(i), 2);
	}
	lua_pushnumber(L, sqrt(d));
	return 1;
}

static int l_trace(lua_State* L)
{
	LUA_PREAMBLE(Matrix, m, 1);

	double d = 0;
	for(int i=0; i<4; i++)
	{
		d += m->component(i,i);
	}
	lua_pushnumber(L, d);
	return 1;
}



static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Matrix::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LuaBaseObject::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"__tostring", l_tostring},
		{"__add",      l_add},
		{"__sub",      l_sub},
		{"__unm",      l_unm},
		{"__mul",      l_mul},
		{"__eq",       l_eq},
		{"set",        l_set},
		{"get",        l_get},
		{"det",        l_det},
		{"makeZero",       l_zero},
		{"makeIdentity",   l_identity},
		{"makeTranslation",l_trans},
		{"makeScale",      l_scale},
		{"makeRotation",   l_rotate},
		{"invert",     l_invert},
		{"inverse",     l_inverse},
		{"list",       l_list},
		{"row",        l_getrow},
		{"col",        l_getcol},
		{"frobeniusNorm", l_frobnorm},
		{"trace", l_trace},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}

