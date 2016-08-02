#include <math.h>
#include <stdio.h>
extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}


#ifndef M_PI
#define M_PI 3.1415926535897932385
#endif

// JOURNAL OF GEOPHYSICAL RESEARCH, VOL. 98, NO. B6, PP. 9551-9555, 1993

// portland group does strange non-compatible long double math. 
#ifndef __PGI
#define ATAN atanl
#define SQRT sqrtl
#define LOG  logl
#define DOUBLE long double
#else
#define ATAN atan
#define SQRT sqrt
#define LOG  log
#define DOUBLE double
#endif




static DOUBLE invtan(const DOUBLE num, const DOUBLE denom)
{
	if(denom == 0)
	{
		if(num > 0)
			return M_PI * 0.5;
		return -M_PI * 0.5;
	}
	const DOUBLE v = ATAN(num/denom);
// 	if(isnan(v))
// // 		printf("%Lf  %Lf\n", num, denom);
	return v;
}

static DOUBLE sign(const DOUBLE x)
{
	if(x > 0)
		return  1;
	if(x < 0)
		return -1;
	return 0;
}

static DOUBLE phi(const DOUBLE n, const DOUBLE d)
{
	DOUBLE v =  0;
	if(n == 0)
		v = 0;
	else
	{
		if(d == 0)
			v = 1000 * sign(n); //arcsinh(10^500) ~= 1000. 10^500 is like infinity
		else
		{
			const DOUBLE p = n / d;
			const DOUBLE t = p + SQRT(1.0 + p*p);
			if(t == 0)
			{
				v = 1000 * sign(n);
			}
			else
				v = LOG(t);
		}
	}
	return v;
}


static DOUBLE g(const DOUBLE x, const DOUBLE y, const DOUBLE z)
{
	const DOUBLE xx = x*x;
	const DOUBLE yy = y*y;
	const DOUBLE zz = z*z;
	const DOUBLE R = SQRT(xx+yy+zz);

	return 	  (x*y*z) * phi(z, SQRT(xx+yy))
			+ (y/6.0) * (3.0*zz-yy) * phi(x, SQRT(yy+zz))
			+ (x/6.0) * (3.0*zz-xx) * phi(y, SQRT(xx+zz))
			- ( z*zz/6.0) * invtan(x*y,(z*R))
			- ( z*yy/2.0) * invtan(x*z,(y*R))
			- ( z*xx/2.0) * invtan(y*z,(x*R))
			- (x*y*R/3.0);
}

static DOUBLE G2(const DOUBLE X,   const DOUBLE Y,   const DOUBLE Z) 
{
	return g(X,Y,Z) - g(X,Y,0);
}
static DOUBLE G1(const DOUBLE X,   const DOUBLE Y,   const DOUBLE Z,
		 const DOUBLE dx2, const DOUBLE dy2, const DOUBLE dz2) 
{
	return G2(X+dx2,Y,Z+dz2) 
		 - G2(X+dx2,Y,Z    )
		 - G2(X,    Y,Z+dz2) 
		 + G2(X,    Y,Z    );
}

static DOUBLE G(const DOUBLE X,   const DOUBLE Y,   const DOUBLE Z,
		 const DOUBLE dx1, const DOUBLE dy1, const DOUBLE dz1, 
		 const DOUBLE dx2, const DOUBLE dy2, const DOUBLE dz2) 
{
	return G1(X,Y,    Z,    dx2,dy2,dz2) 
		 - G1(X,Y-dy1,Z,    dx2,dy2,dz2) 
		 - G1(X,Y,    Z-dz1,dx2,dy2,dz2) 
		 + G1(X,Y-dy1,Z-dz1,dx2,dy2,dz2);
}





static DOUBLE f(const DOUBLE x, const DOUBLE y, const DOUBLE z)
{
	const DOUBLE xx = x*x;
	const DOUBLE yy = y*y;
	const DOUBLE zz = z*z;
	const DOUBLE R = SQRT(xx+yy+zz);
	
	return 	  y * 0.5 * (zz-xx) * phi(y,SQRT(xx+zz))
			+ z * 0.5 * (yy-xx) * phi(z,SQRT(xx+yy))
			- x*y*z * invtan(y*z, x*R)
			+ (1.0/6.0) * (2.0*xx-yy-zz) * R;
}

static DOUBLE F2(const DOUBLE X,   const DOUBLE Y,   const DOUBLE Z,
		 const DOUBLE dx1, const DOUBLE dy1, const DOUBLE dz1, 
		 const DOUBLE dx2, const DOUBLE dy2, const DOUBLE dz2) 
{
	const DOUBLE v = 
	       f(X,Y,Z) - f(0,Y,Z) - f(X,0,Z) + f(X,Y,0);
	return v;
}

static DOUBLE F1(const DOUBLE X,   const DOUBLE Y,   const DOUBLE Z,
		 const DOUBLE dx1, const DOUBLE dy1, const DOUBLE dz1, 
		 const DOUBLE dx2, const DOUBLE dy2, const DOUBLE dz2) 
{
	const DOUBLE v =
	        F2(X, Y,     Z,     dx1, dy1, dz1, dx2, dy2, dz2)
	      - F2(X, Y-dy1, Z,     dx1, dy1, dz1, dx2, dy2, dz2)
		  - F2(X, Y,     Z-dz1, dx1, dy1, dz1, dx2, dy2, dz2)
		  + F2(X, Y-dy1, Z-dz1, dx1, dy1, dz1, dx2, dy2, dz2);
	return v;
}

static DOUBLE F(const DOUBLE X,   const DOUBLE Y,   const DOUBLE Z,
		 const DOUBLE dx1, const DOUBLE dy1, const DOUBLE dz1, 
		 const DOUBLE dx2, const DOUBLE dy2, const DOUBLE dz2) 
{
	const DOUBLE v = 
	         F1(X, Y+dy2, Z+dz2, dx1, dy1, dz1, dx2, dy2, dz2) 
		   - F1(X, Y,     Z+dz2, dx1, dy1, dz1, dx2, dy2, dz2)  
		   - F1(X, Y+dy2, Z,     dx1, dy1, dz1, dx2, dy2, dz2)  
		   + F1(X, Y,     Z,     dx1, dy1, dz1, dx2, dy2, dz2);	
		   
	return v;
}

static void shuffle(double* dest, const double* src, const int* o)
{
	dest[0] = src[o[0]];
	dest[1] = src[o[1]];
	dest[2] = src[o[2]];
}


double magnetostatic_Nxx(const double X,   const double Y,   const double Z,
				  const double* target, const double* source)
{
	const double* p1 = target;
	const double* p2 = source;
	const DOUBLE dx1 = p1[0];
	const DOUBLE dx2 = p2[0];
	const DOUBLE v = p1[0]*p1[1]*p1[2];
	
	const DOUBLE FF1 = F(X,         Y, Z, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
	const DOUBLE FF2 = F(X+dx2-dx1, Y, Z, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
	const DOUBLE FF3 = F(X-dx1,     Y, Z, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
	const DOUBLE FF4 = F(X+dx2,     Y, Z, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);

	return -(1.0/(4.0*M_PI * v)) * (FF1 + FF2 - FF3 - FF4);
}


double magnetostatic_Nyy(const double X,   const double Y,   const double Z,
				  const double* target, const double* source)
{
	double t[3], s[3];
	const int o[3] = {1,2,0};
	shuffle(t, target, o);
	shuffle(s, source, o);

	return magnetostatic_Nxx(Y, Z, X, t, s);
}

double magnetostatic_Nzz(const double X,   const double Y,   const double Z,
				  const double* target, const double* source)
{
	double t[3], s[3];
	const int o[3] = {2,0,1};
	shuffle(t, target, o);
	shuffle(s, source, o);

	return magnetostatic_Nxx(Z, X, Y, t, s);
}


double magnetostatic_Nxy(const double X,   const double Y,   const double Z,
				  const double* target, const double* source)
{
	const double* p1 = target;
	const double* p2 = source;
	const DOUBLE dx1 = p1[0];
	const DOUBLE dx2 = p2[0];
	const DOUBLE dy1 = p1[1];
	const DOUBLE dy2 = p2[1];
	const DOUBLE v = p1[0]*p1[1]*p1[2];
	return -(1.0 / (4.0*M_PI*v)) * (
		  G(X,     Y,     Z, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]) 
		- G(X-dx1, Y,     Z, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]) 
		- G(X,     Y+dy2, Z, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]) 
		+ G(X-dx1, Y+dy2, Z, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]));
}


double magnetostatic_Nxz(const double X,   const double Y,   const double Z,
				  const double* target, const double* source)
{
	double t[3], s[3];
	const int o[3] = {0,2,1};
	shuffle(t, target, o);
	shuffle(s, source, o);
	return magnetostatic_Nxy(X, Z, Y, t, s);
}

double magnetostatic_Nzx(const double X,   const double Y,   const double Z,
				  const double* target, const double* source)
{
	double t[3], s[3];
	const int o[3] = {2,0,1};
	
	shuffle(t, target, o);
	shuffle(s, source, o);
	return magnetostatic_Nxy(Z, X, Y, t, s);
}

double magnetostatic_Nyx(const double X,   const double Y,   const double Z,
				  const double* target, const double* source)
{
	double t[3], s[3];
	const int o[3] = {1,0,2};
	shuffle(t, target, o);
	shuffle(s, source, o);
	return magnetostatic_Nxy(Y, X, Z, t, s);
}

double magnetostatic_Nyz(const double X,   const double Y,   const double Z,
				  const double* target, const double* source)
{
	double t[3], s[3];
	const int o[3] = {1,2,0};
	shuffle(t, target, o);
	shuffle(s, source, o);
	return magnetostatic_Nxy(Y, Z, X, t, s);
}


double magnetostatic_Nzy(const double X,   const double Y,   const double Z,
				  const double* target, const double* source)
{
	double t[3], s[3];
	const int o[3] = {2,1,0};
	shuffle(t, target, o);
	shuffle(s, source, o);
	return magnetostatic_Nxy(Z, Y, X, t, s);
}













#define PROTO(func) static int l_ ## func(lua_State* L) {

#define FUNC1(func) \
PROTO(func) \
	const DOUBLE v1 = lua_tonumber(L, 1); \
	lua_pushnumber(L,  func(v1) ); return 1; }

#define FUNC2(func) \
PROTO(func) \
	const DOUBLE v1 = lua_tonumber(L, 1); \
	const DOUBLE v2 = lua_tonumber(L, 2); \
	lua_pushnumber(L,  func(v1,v2) ); return 1; }

#define FUNC3(func) \
PROTO(func) \
	const DOUBLE v1 = lua_tonumber(L, 1); \
	const DOUBLE v2 = lua_tonumber(L, 2); \
	const DOUBLE v3 = lua_tonumber(L, 3); \
	lua_pushnumber(L,  func(v1,v2,v3) ); return 1; }

#define FUNC6(func) \
PROTO(func) \
	const DOUBLE v1 = lua_tonumber(L, 1); \
	const DOUBLE v2 = lua_tonumber(L, 2); \
	const DOUBLE v3 = lua_tonumber(L, 3); \
	const DOUBLE v4 = lua_tonumber(L, 4); \
	const DOUBLE v5 = lua_tonumber(L, 5); \
	const DOUBLE v6 = lua_tonumber(L, 6); \
	lua_pushnumber(L,  func(v1,v2,v3,v4,v5,v6) ); return 1; }

#define FUNC9(func) \
PROTO(func) \
	const DOUBLE v1 = lua_tonumber(L, 1); \
	const DOUBLE v2 = lua_tonumber(L, 2); \
	const DOUBLE v3 = lua_tonumber(L, 3); \
	const DOUBLE v4 = lua_tonumber(L, 4); \
	const DOUBLE v5 = lua_tonumber(L, 5); \
	const DOUBLE v6 = lua_tonumber(L, 6); \
	const DOUBLE v7 = lua_tonumber(L, 7); \
	const DOUBLE v8 = lua_tonumber(L, 8); \
	const DOUBLE v9 = lua_tonumber(L, 9); \
	lua_pushnumber(L,  func(v1,v2,v3,v4,v5,v6,v7,v8,v9) ); return 1; }



FUNC1(sign)
FUNC2(invtan)
FUNC2(phi)
FUNC3(g)
FUNC3(f)
FUNC3(G2)
FUNC6(G1)
FUNC9(G)
FUNC9(F)
FUNC9(F1)
FUNC9(F2)

#define REG(func) lua_pushstring(L, #func); lua_pushcfunction(L, l_##func); lua_settable(L, -3);

void register_mag3d_internal_functions(lua_State* L)
{
	lua_getglobal(L, "Magnetostatics3D");
	
	REG(sign);
	REG(invtan);
	REG(phi);
	REG(g);
	REG(f);
	REG(G1);
	REG(G2);
	REG(G);
	REG(F1);
	REG(F2);
	REG(F);
	
	lua_pop(L, 1);
}

