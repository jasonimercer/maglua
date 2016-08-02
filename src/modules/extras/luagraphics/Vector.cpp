#include <math.h>
#include "Vector.h"
#include <stdlib.h>
#include <vector>
#include <string.h>
using namespace std;

Vector::Vector(double x, double y, double z)
	: LuaBaseObject(hash32("Vector"))
{
	_xyz[0] = x;
	_xyz[1] = y;
	_xyz[2] = z;
	_xyz[3] = 0;
}

int Vector::luaInit(lua_State* L)
{
	for(int i=0; i<3; i++)
	{
		if(lua_isnumber(L, i+1))
			_xyz[i] = lua_tonumber(L, i+1);
	}
	_xyz[3] = 0;
	return 0;
}

void Vector::push(lua_State* L)
{
	luaT_push<Vector>(L, this);
}

Vector::Vector(const double* r3)
	: LuaBaseObject(hash32("Vector"))
{
	_xyz[0] = r3[0];
	_xyz[1] = r3[1];
	_xyz[2] = r3[2];
	_xyz[3] = 0;
}

Vector::Vector(const Vector* other)
	: LuaBaseObject(hash32("Vector"))
{
	*this = *other;
}

Vector::Vector(const Vector& other)
	: LuaBaseObject(hash32("Vector"))
{
	*this = other;
}

Vector::~Vector()
{
}

void Vector::swap(Vector& v2)
{
	double f;
	for(int i=0; i<3; i++)
	{
		f = component(i);
		setComponent(i, v2.component(i));
		v2.setComponent(i, f);
	}
}

#include <stdio.h>
void Vector::setComponent(unsigned int c, double value)
{
	if(c < 0)
		c = 0;
	if(c > 3)
		c = 3;
// 	_xyz[c&3] = value;
	_xyz[c] = value;
}

void Vector::setX(double v)
{
	_xyz[0] = v;
}

void Vector::setY(double v)
{
	_xyz[1] = v;	
}

void Vector::setZ(double v)
{
	_xyz[2] = v;	
}

void Vector::set(double x, double y, double z)
{
	setX(x);
	setY(y);
	setZ(z);
}

void Vector::set(const double* v3)
{
	setX(v3[0]);
	setY(v3[1]);
	setZ(v3[2]);
}


static double randf()
{
	int i = rand();
	double r = (double)i / (double)RAND_MAX;
	
	return r;
}


static void randN(double* n1, double* n2)
{
	double x, y, r;
	do {
		x = randf() * 2.0 - 1.0;
		y = randf() * 2.0 - 1.0;
		r = x*x + y*y;
	}while (r == 0.0 || r > 1.0);
	
	double d = sqrt(-2.0*log(r)/r);
	*n1 = x*d;
	*n2 = y*d;
}


void Vector::bmRandom(double scale) //box-muller random normal coordinates
{
	double a, b;
	randN(&a, &b);
	
	setX(scale * a);
	setY(scale * b);
	randN(&a, &b);
	setZ(scale * a);
}

void Vector::randomize(double scale)
{
	for(int i=0; i<3; i++)
	{
		setComponent(i, (randf() * 4.0 - 2.0) * component(i) + randf()*2.0*scale-scale);
	}
}

Vector Vector::min(const Vector& a, const Vector& b)
{
	Vector v = a;
	for(int i=0; i<3; i++)
	{
		const double x = b.component(i);
		if(x < v.component(i))
			v.setComponent(i, x);
	}
	return v;
}

Vector Vector::max(const Vector& a, const Vector& b)
{
	Vector v = a;
	for(int i=0; i<3; i++)
	{
		const double x = b.component(i);
		if(x > v.component(i))
			v.setComponent(i, x);
	}
	return v;
}


	
double Vector::component(unsigned int c) const
{
	if(c > 3)
		return component(3);
	return _xyz[c];
}

double Vector::x() const
{
	return _xyz[0];
}

double Vector::y() const
{	
	return _xyz[1];
}

double Vector::z() const
{
	return _xyz[2];
}

void Vector::normalize(double unity)
{
	double len = length();
	if(len == 0)
		*this = Vector(unity,0,0);
	else
	{
		*this *= (unity/len);
	}
}



void Vector::clamp(double maxlength)
{
	if(lengthSquared() > maxlength*maxlength)
		normalize(maxlength);
}

Vector Vector::normalized(double unity) const
{
	Vector v(*this);
	v.normalize(unity);
	return v;
}

double Vector::length() const
{
	return sqrt(lengthSquared());
}

double Vector::lengthSquared() const
{
	return dot(*this);
}

const double* Vector::vec() const
{
	return _xyz;
}

double Vector::distanceToPlane(const Vector& n, const Vector& x)
{
	return n.dot(*this - x);
}

double Vector::dot(const Vector& rhs) const
{
	const double* a = _xyz;
	const double* b = rhs.vec();
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

Vector Vector::cross(const Vector& rhs) const
{
	return Vector::crossProduct(*this, rhs);
}

double Vector::radiansBetween(const Vector& a, const Vector& b)
{
	if(a.lengthSquared() == 0 || b.lengthSquared() == 0)
		return 0;
	
	return acos(a.dot(b) / (a.length() * b.length()));
}


Vector Vector::crossProduct(const Vector& a, const Vector& b)
{
	Vector c;
	c.setX(a.y()*b.z() - a.z()*b.y());
	c.setY(a.z()*b.x() - a.x()*b.z());
	c.setZ(a.x()*b.y() - a.y()*b.x());
	return c;
}

double Vector::dotProduct(const Vector& a, const Vector& b)
{
	return a.dot(b);
}



Vector& Vector::operator=(const Vector& rhs)
{
	for(int i=0; i<3; i++)
		setComponent(i, rhs.component(i));
	return *this;
}


Vector& Vector::operator+=(const Vector &rhs)
{
	for(int i=0; i<3; i++)
		setComponent(i, component(i) + rhs.component(i));
	return *this;
}

Vector& Vector::operator-=(const Vector &rhs)
{
	for(int i=0; i<3; i++)
	{
		double x = component(i) - rhs.component(i);
		setComponent(i, x);
	}
	return *this;
}

Vector& Vector::operator*=(const double value)
{
	for(int i=0; i<3; i++)
		setComponent(i, value * component(i));
	return *this;
}

Vector& Vector::operator/=(const double value)
{
	double ivalue = 1.0 / value;
	for(int i=0; i<3; i++)
		setComponent(i, component(i) * ivalue);
	return *this;
}

const Vector Vector::operator+(const Vector &other) const
{
    Vector result = *this;
    result += other;
    return result;
}

const Vector Vector::operator-(const Vector &other) const
{
	Vector result = *this;
	result -= other;
	return result;
}

const Vector Vector::operator-() const
{
	Vector result = *this;
	return -1.0*result;
}

const Vector operator*(const double m, const Vector& v)
{
	return v*m;
}

const Vector Vector::operator*(const double value) const
{
	Vector result = *this;
	result *= value;
	return result;
}

const Vector Vector::operator/(const double value) const
{
	Vector result = *this;
	result /= value;
	return result;
}

bool Vector::operator==(const Vector &other) const
{
	for(int i=0; i<3; i++)
		if(component(i) != other.component(i))
			return false;
	return true;
}

bool Vector::operator!=(const Vector &other) const
{
    return !(*this == other);
}


ostream& operator<<(ostream& out, const Vector& v)
{
	out << v.x() << ", " << v.y() << ", " << v.z();
	return out;
}


void Vector::zero()
{
	setX(0);
	setY(0);
	setZ(0);
}

// n plane normal
// x (any) point on plane
void Vector::projectOntoPlane(const Vector& n)
{
	*this = projectedOntoPlane(n);
}

// n plane normal
// x (any) point on plane
Vector Vector::projectedOntoPlane(const Vector& n) const
{
	Vector a = *this;
	Vector b = Vector::crossProduct(n, a);
	Vector c = Vector::crossProduct(b, n);
	return c;
}


void Vector::projectOnto(const Vector& v)
{
	if(v.lengthSquared() == 0)
	{
		zero();
	}
	else
	{
		const double p = dot(v) / v.dot(v);
// 		*this = v;
		*this = (v * p);
// 		set(v);
// 		mult(p);
	}
}

void Vector::rotateAbout(const Vector& vec, double theta)
{
	Vector uvw = vec.normalized();
	
	double _x = x();
	double _y = y();
	double _z = z();
	double ux_vy_wz;
	
	double _u = uvw.x();
	double _v = uvw.y();
	double _w = uvw.z();
	
	double cost = cos(theta);
	double sint = sin(theta);
	ux_vy_wz =  _u*_x+_v*_y+_w*_z;
	
	setX(_u*(ux_vy_wz)+(_x*(_v*_v+_w*_w)-_u*(_v*_y+_w*_z))*cost + (-_w*_y+_v*_z)*sint);
	setY(_v*(ux_vy_wz)+(_y*(_u*_u+_w*_w)-_v*(_u*_x+_w*_z))*cost + ( _w*_x-_u*_z)*sint);
	setZ(_w*(ux_vy_wz)+(_z*(_u*_u+_v*_v)-_w*(_u*_x+_v*_y))*cost + (-_v*_x+_u*_y)*sint);
}




int lua_makevector(lua_State* L, int idx, Vector* v)
{
	return lua_makevector(L, idx, *v);
}

int lua_makevector(lua_State* L, int idx, Vector& v)
{
	if(luaT_is<Vector>(L, idx))
	{
		Vector* b = luaT_to<Vector>(L, idx);
		if(b)
		{
			v = *b;
		}
		return 1;
	}
	
	if(lua_istable(L, idx))
	{
		for(int i=0; i<3; i++)
		{
			lua_pushinteger(L, 1+i);
			lua_gettable(L, idx);
			v.setComponent(i, lua_tonumber(L, -1));
			lua_pop(L, 1);
		}
		return 1;
	}
	
	for(int i=0; i<3; i++)
	{
		v.setComponent(i, lua_tonumber(L, i+idx));
	}
	return 3;
}

static int l_vector_tostring(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);

	lua_pushfstring(L, "%f, %f, %f", v->x(), v->y(), v->z());
	return 1;
}

static int l_vector_add(lua_State* L)
{
	LUA_PREAMBLE(Vector, a, 1);
	LUA_PREAMBLE(Vector, b, 2);

	luaT_push<Vector>(L, new Vector(*a + *b));
	return 1;
}
static int l_vector_sub(lua_State* L)
{
	LUA_PREAMBLE(Vector, a, 1);
	LUA_PREAMBLE(Vector, b, 2);

	luaT_push<Vector>(L, new Vector(*a - *b));
	return 1;
}
static int l_vector_unm(lua_State* L)
{
	LUA_PREAMBLE(Vector, a, 1);
	
	luaT_push<Vector>(L, new Vector(-1 * (*a)));
	return 1;
}
static int l_vector_mul(lua_State* L)
{
	LUA_PREAMBLE(Vector, a, 1);
	
	luaT_push<Vector>(L, new Vector((*a) * lua_tonumber(L, 2)));
	return 1;
}
static int l_vector_eq(lua_State* L)
{
	LUA_PREAMBLE(Vector, a, 1);
	LUA_PREAMBLE(Vector, b, 2);

	lua_pushboolean(L, (*a) == (*b));
	return 1;
}


static int l_vector_get(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);

	int c = lua_tointeger(L, 2) - 1;
	lua_pushnumber(L, v->component(c));
	
	return 1;	
}


static int l_vector_set(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);
	
	Vector a;
	lua_makevector(L, 2, a);
	
	*v = a;
	return 0;
}

static int l_vector_setX(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);
	v->setX(lua_tonumber(L, 2));
	return 0;
}

static int l_vector_setY(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);
	v->setY(lua_tonumber(L, 2));
	return 0;
}

static int l_vector_setZ(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);
	v->setZ(lua_tonumber(L, 2));
	return 0;
}

LUAFUNC_GET_DOUBLE(Vector, x(), l_vector_getX)
LUAFUNC_GET_DOUBLE(Vector, y(), l_vector_getY)
LUAFUNC_GET_DOUBLE(Vector, z(), l_vector_getZ)
LUAFUNC_GET_DOUBLE(Vector, length(), l_vector_length)


static int l_vector_dot(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);
	LUA_PREAMBLE(Vector, b, 2);
	lua_pushnumber(L, v->dot(*b));
	return 1;
}

static int l_vector_cross(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);
	LUA_PREAMBLE(Vector, b, 2);
			
	luaT_push<Vector>(L, new Vector(v->cross(*b)));
	return 1;
}

static int l_vector_project(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);
	LUA_PREAMBLE(Vector, b, 2);

	v->projectOnto(*b);
	return 0;
}

static int l_vector_rotate(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);
	LUA_PREAMBLE(Vector, a, 2);
	double theta = lua_tonumber(L, 3);
	
	v->rotateAbout(*a, theta);
	return 0;
}

static int l_vector_normalize(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);
	v->normalize();
	return 0;
}

static int l_vector_normalized(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);

	Vector* v2 = new Vector(*v);
	v2->normalize();
		
	luaT_push<Vector>(L, v2);
	
	return 1;
}

static int l_vector_totable(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);

	lua_newtable(L);
	for(int i=0; i<3; i++)
	{
		lua_pushinteger(L, i+1);
		lua_pushnumber(L, v->component(i));
		lua_settable(L, -3);
	}
	return 1;
}

static void swapi(int* a, int i, int j)
{
	if(i == j) return;
	
	int t = a[i];
	a[i] = a[j];
	a[j] = t;
}

static int reorder(int n, int* a)
{
	int k = -1;
	int l = -1;
	for(int K=0; K<n-1; K++)
	{
		if(a[K] < a[K+1])
			k = K;
	}
	
	if(k < 0)
		return -1;
	
	for(int L=0; L<n; L++)
	{
		if(a[k] < a[L])
			l = L;
	}
	
	//swap a[k] and a[l]
	swapi(a, k, l);

	// reverse from a[k+1] to a[n-1]
	int end_range = n-k-1;
	for(int i=1; i<end_range; i++)
	{
		swapi(a, k+i, n-i);
	}
	
	return k;
}

static double _match_val(vector<Vector*>& v1, vector<Vector*>& v2, int* order)
{
	double sum = 0;
	for(unsigned int i=0; i<v1.size(); i++)
	{
		Vector d = (*v1[i]) - (*v2[order[i]]);
		sum += d.length();
	}
	return sum;
}

static double _vector_match(vector<Vector*>& v1, vector<Vector*>& v2, int* order)
{
	int n1 = v1.size();
	int n2 = v2.size();
	int* current_order = new int[n2];
	double best_match = 1e10;
	
	// order will hold all values
	for(int i=0; i<n2; i++)
	{
		current_order[i] = i;
	}
	
	best_match = _match_val(v1,v2,current_order);
	
	memcpy(order, current_order, sizeof(int)*n1);
	
	int k = 0;
	while(k>=0)
	{
		k = reorder(n2, current_order);
			
		if(k < n1 && k >= 0) //then the reordering shifted something in the first n1 elements
		{
			double current_match = _match_val(v1,v2,current_order);
			
			if(current_match < best_match)
			{
				best_match = current_match;
				memcpy(order, current_order, sizeof(int)*n1);
			}
		}
	}
	
	
	delete [] current_order;
	return best_match;
}

static int l_nearest_pairs(lua_State* L)
{
	if(!lua_istable(L, 1) || !lua_istable(L, 2))
		return luaL_error(L, "two tables of vectors expected");
	
	double max_len = -1;
	if(lua_isnumber(L, 3))
		max_len = lua_tonumber(L, 3);
	
	vector<int> k[2];
	vector<Vector*> v[2];
//	int* ordering;

	int small = 0;
	int big   = 1;
//	double fit;

	for(int i=0; i<2; i++)
	{
		lua_pushnil(L);
		while(lua_next(L, i+1))
		{
			Vector* vec = luaT_to<Vector>(L, -1);
			if(vec)
			{
				lua_pushvalue(L, -2); //dup key
				k[i].push_back(luaL_ref(L, LUA_REGISTRYINDEX));
				v[i].push_back(vec);
			}
			lua_pop(L, 1);
		}
	}
	
		
	if(v[0].size() > v[1].size())
	{
		small = 1;
		big = 0;
	}
	
	// for each element in the small list, find the closest in the big list
	
	double sum = 0;
	lua_newtable(L);
	
	for(unsigned int i=0; i<v[small].size(); i++)
	{
		
		double bestv = (*v[small][i] - *v[big][0]).length();
		int bestj = 0;
		for(unsigned int j=1; j<v[big].size(); j++)
		{
			double a = (*v[small][i] - *v[big][j]).length();
			if(a < bestv)
			{
				bestv = a;
				bestj = j;
			}
		}
		if(bestv < max_len || max_len < 0)
		{
			sum += bestv;
			lua_pushinteger(L, i+1);

			lua_newtable(L);
			lua_pushinteger(L, 1);
			lua_rawgeti(L, LUA_REGISTRYINDEX, k[small][i]);
			lua_settable(L, -3);
			lua_pushinteger(L, 2);
			lua_rawgeti(L, LUA_REGISTRYINDEX, k[big][bestj]);
			lua_settable(L, -3);

			lua_settable(L, -3);
		}
		else
			sum += max_len;
	}
	
	lua_pushnumber(L, sum);
	lua_pushvalue(L, -2); //make val 1st return arg
	
	for(int i=0; i<2; i++)
	{
		for(unsigned int j=0; j<k[i].size(); j++)
		{
			luaL_unref(L, LUA_REGISTRYINDEX, k[i][j]);			
		}
	}
	
	return 2;
}

// input is 2 tables of vectors
// goal is to return the best matching based on distance
static int l_match_groups(lua_State* L)
{
	if(!lua_istable(L, 1) || !lua_istable(L, 2))
		return luaL_error(L, "two tables of vectors expected");
	
	vector<int> k[2];
	vector<Vector*> v[2];
	int* ordering;

	int small = 0;
	int big   = 1;
	double fit;

	for(int i=0; i<2; i++)
	{
		lua_pushnil(L);
		while(lua_next(L, i+1))
		{
			Vector* vec = luaT_to<Vector>(L, -1);
			if(vec)
			{
				lua_pushvalue(L, -2); //dup key
				k[i].push_back(luaL_ref(L, LUA_REGISTRYINDEX));
				v[i].push_back(vec);
			}
			lua_pop(L, 1);
		}
	}

	
	ordering = new int[v[0].size()]; //at least as big as smallest
	
	// now have 2 sets of vectors in v[0] and v[1]. 
	// need to find best match between groups
	
	if(v[0].size() > v[1].size())
	{
		small = 1;
		big = 0;
	}
	
	fit = _vector_match(v[small], v[big], ordering);
	
	lua_newtable(L);
	for(unsigned int i=0; i<v[small].size(); i++)
	{
		lua_pushinteger(L, i+1);
		
		lua_newtable(L);
		lua_pushinteger(L, small+1);
		lua_rawgeti(L, LUA_REGISTRYINDEX, k[small][i]);
		lua_settable(L, -3);

		lua_pushinteger(L, big+1);
		lua_rawgeti(L, LUA_REGISTRYINDEX, k[big][ ordering[i] ]);
		lua_settable(L, -3);
		
		lua_settable(L, -3);
	}
	
	for(int a=0; a<2; a++)
	{
		for(unsigned int i=0; i<k[a].size(); i++)
		{
			luaL_unref(L, LUA_REGISTRYINDEX, k[a][i]);
		}
	}
	
	delete [] ordering;
	
	lua_pushnumber(L, fit);
	
	return 2;
}

static int l_s2c(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);
	
	const double t = v->x();
	const double p = v->y();
	const double r = v->z();
	
	const double x = r * cos(t) * sin(p);
	const double y = r * sin(t) * sin(p);
	const double z = r * cos(p);
	
	v->setX(x);
	v->setY(y);
	v->setZ(z);
	return 0;
}

static int l_c2s(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);

	const double x = v->x();
	const double y = v->y();
	const double z = v->z();
	
	const double r = sqrt(x*x+y*y+z*z);
	const double t = atan2(y,x);
	double p = 0;
	if(r > 0)
	{
		p = acos(z/r);
	}
		
	v->setX(t);
	v->setY(p);
	v->setZ(z);
	return 0;
}

static int l_scale(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);

	if(lua_isnumber(L, 2) && !lua_isnumber(L, 3))
	{
		(*v)*=lua_tonumber(L, 2);
		return 0;
	}

	Vector vec;
	lua_makevector(L, 2, vec);
	v->setX(v->x() * vec.x());
	v->setY(v->y() * vec.y());
	v->setZ(v->z() * vec.z());
	return 0;
}

static int l_scaled(lua_State* L)
{
	LUA_PREAMBLE(Vector, v, 1);

	Vector* ww = new Vector(*v);

	if(lua_isnumber(L, 2) && !lua_isnumber(L, 3))
	{
		(*ww)*=lua_tonumber(L, 2);
		luaT_push<Vector>(L, ww);
		return 1;
	}


	Vector vec;
	lua_makevector(L, 2, vec);
	ww->setX(v->x() * vec.x());
	ww->setY(v->y() * vec.y());
	ww->setZ(v->z() * vec.z());
	
	luaT_push<Vector>(L, ww);
	return 1;
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* Vector::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, LuaBaseObject::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"__tostring",   l_vector_tostring},
		{"__add",        l_vector_add},
		{"__sub",        l_vector_sub},
		{"__unm",        l_vector_unm},
		{"__mul",        l_vector_mul},
		{"__eq",         l_vector_eq},
		{"set",          l_vector_set},
		{"get",          l_vector_get},
		{"setX",         l_vector_setX},
		{"setY",         l_vector_setY},
		{"setZ",         l_vector_setZ},
		{"x",            l_vector_getX},
		{"y",            l_vector_getY},
		{"z",            l_vector_getZ},
		{"length",       l_vector_length},
		{"cross",        l_vector_cross},
		{"dot",          l_vector_dot},
		{"projectOnto",  l_vector_project},
		{"rotateAbout",  l_vector_rotate},
		{"normalize",    l_vector_normalize},
		{"normalized",   l_vector_normalized},
		{"toTable",      l_vector_totable},
		{"s2c",          l_s2c},
		{"c2s",          l_c2s},
		{"scale",        l_scale},
		{"scaled",       l_scaled},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}

