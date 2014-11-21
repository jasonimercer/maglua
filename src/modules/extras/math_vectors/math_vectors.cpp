#include "math_vectors.h"
#include <math.h>
#include "math_vectors_luafuncs.h"
#include "luabaseobject.h"

#include <vector>
#include <deque>
using namespace std;
#include "info.h"


static int l_erfc(lua_State* L)
{
	if(lua_isnumber(L, 1))
	{
		double x = lua_tonumber(L, 1);
		lua_pushnumber(L, erfc(x));
		return 1;
	}
	return 0;
}
static int l_erf(lua_State* L)
{
	if(lua_isnumber(L, 1))
	{
		double x = lua_tonumber(L, 1);
		lua_pushnumber(L, erf(x));
		return 1;
	}
	return 0;
}

struct p3
{
    p3() {set(0,0,0);}
    p3(const p3& p) {set(p.x);}
    p3(double a, double b, double c) {set(a,b,c);}
    p3(const double* p) {set(p);}
    void set(const double* v) {set(v[0],v[1],v[2]);}
    void set(double a, double b, double c)
	{x[0] = a; x[1] = b; x[2] = c;}
    void normalize()
	{   double len = sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
	    x[0] /= len; x[1] /= len; x[2] /= len;}
		
    double x[3];
};
struct t3
{
    t3(int a, int b, int c) {set(a,b,c);}
    t3(const t3& t) {set(t.p[0], t.p[1], t.p[2]);}
    void set(int a, int b, int c)
	{p[0] = a; p[1] = b; p[2] = c;}
    int p[3];
};
// generate unit vectors evenly spaced on a sphere
// done via sub-divisions of a tetrahedron
// (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1)
static int l_vectorsicosphere(lua_State* L)
{
    int n = lua_tointeger(L, 1);
    vector<p3> pts;
    deque<t3> tris;

/*
    pts.push_back(p3( 1, 1, 1));
    pts.push_back(p3( 1,-1,-1));
    pts.push_back(p3(-1, 1,-1));
    pts.push_back(p3(-1,-1, 1));
*/

    pts.push_back(p3( 0, 0, sqrt(2.0/3.0) - 0.5 / sqrt(6.0)));
    pts.push_back(p3( -1 / sqrt(12.0), -0.5, -0.5/sqrt(6.0)));
    pts.push_back(p3( -1 / sqrt(12.0),  0.5, -0.5/sqrt(6.0)));
    pts.push_back(p3(  1 / sqrt(3.0),  0.0, -0.5/sqrt(6.0)));

    for(int i=0; i<4; i++)
	pts[i].normalize();

    tris.push_back( t3(1,2,3) );
    tris.push_back( t3(0,1,2) );
    tris.push_back( t3(0,1,3) );
    tris.push_back( t3(0,2,3) );

    
    while(pts.size() < n)
    {
	t3& t = tris.front();

	double d[3] = {0,0,0};//center

	const double* a = pts[ t.p[0] ].x;
	const double* b = pts[ t.p[1] ].x;
	const double* c = pts[ t.p[2] ].x;

	d[0] = (a[0] + b[0] + c[0] ) / 3.0;
	d[1] = (a[1] + b[1] + c[1] ) / 3.0;
	d[2] = (a[2] + b[2] + c[2] ) / 3.0;

	pts.push_back( p3(d) );
	pts.back().normalize();
	int q = pts.size() - 1;

	tris.push_back( t3(t.p[0], t.p[1], q ));
	tris.push_back( t3(t.p[1], t.p[2], q ));
	tris.push_back( t3(t.p[2], t.p[0], q ));
	tris.pop_front();
    }

    lua_newtable(L);
    for(int i=0; i<n; i++)
    {
	lua_pushinteger(L, i+1);
	lua_newtable(L);
	for(int j=0; j<3; j++)
	{
	    lua_pushinteger(L, j+1);
	    lua_pushnumber(L, pts[i].x[j]);
	    lua_settable(L, -3);
	}
	lua_settable(L, -3);
    }

    return 1;
}

extern "C"
{
MATH_VECTORS_API int lib_register(lua_State* L)
{
	lua_getglobal(L, "math");
	lua_pushstring(L, "erfc");
	lua_pushcfunction(L, l_erfc);
	lua_settable(L, -3);

	lua_pushstring(L, "erf");
	lua_pushcfunction(L, l_erf);
	lua_settable(L, -3);

	lua_pushstring(L, "vectorIcosphere");
	lua_pushcfunction(L, l_vectorsicosphere);
	lua_settable(L, -3);

	lua_pop(L, 1); // math table

        luaL_dofile_math_vectors_luafuncs(L);
	
	return 0;
}

MATH_VECTORS_API int lib_version(lua_State* L)
{
	return __revi;
}

MATH_VECTORS_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Math-Vectors";
#else
	return "Math-Vectors-Debug";
#endif
}

MATH_VECTORS_API int lib_main(lua_State* L)
{
	return 0;
}
}
