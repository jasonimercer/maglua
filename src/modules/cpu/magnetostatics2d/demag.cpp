#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.1415926535897932385
#endif

// JOURNAL OF GEOPHYSICAL RESEARCH, VOL. 98, NO. B6, PP. 9551-9555, 1993

#if 1
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

DOUBLE invtan(const DOUBLE num, const DOUBLE denom)
{
	if(denom == 0)
	{
		if(num > 0)
			return M_PI * 0.5;
		return -M_PI * 0.5;
	}
	return ATAN(num/denom);
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
	if(n == 0)
		return 0;
	if(d == 0)
		return 1000 * sign(n); //arcsinh(10^500) ~= 1000. 10^500 is like infinity
	const DOUBLE p = n / d;
	return LOG(p + SQRT(1.0 + p*p));
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
	return f(X,Y,Z) - f(0,Y,Z) - f(X,0,Z) + f(X,Y,0);
}

static DOUBLE F1(const DOUBLE X,   const DOUBLE Y,   const DOUBLE Z,
		 const DOUBLE dx1, const DOUBLE dy1, const DOUBLE dz1, 
		 const DOUBLE dx2, const DOUBLE dy2, const DOUBLE dz2) 
{
	return  F2(X, Y,     Z,     dx1, dy1, dz1, dx2, dy2, dz2)
	      - F2(X, Y-dy1, Z,     dx1, dy1, dz1, dx2, dy2, dz2)
		  - F2(X, Y,     Z-dz1, dx1, dy1, dz1, dx2, dy2, dz2)
		  + F2(X, Y-dy1, Z-dz1, dx1, dy1, dz1, dx2, dy2, dz2);
}

static DOUBLE F(const DOUBLE X,   const DOUBLE Y,   const DOUBLE Z,
		 const DOUBLE dx1, const DOUBLE dy1, const DOUBLE dz1, 
		 const DOUBLE dx2, const DOUBLE dy2, const DOUBLE dz2) 
{
	return   F1(X, Y+dy2, Z+dz2, dx1, dy1, dz1, dx2, dy2, dz2) 
		   - F1(X, Y,     Z+dz2, dx1, dy1, dz1, dx2, dy2, dz2)  
		   - F1(X, Y+dy2, Z,     dx1, dy1, dz1, dx2, dy2, dz2)  
		   + F1(X, Y,     Z,     dx1, dy1, dz1, dx2, dy2, dz2);	
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
	return (1.0/(4.0*M_PI * v)) * (
		  F(X,         Y, Z, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]) 
		+ F(X+dx2-dx1, Y, Z, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]) 
		- F(X-dx1,     Y, Z, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]) 
		- F(X+dx2,     Y, Z, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])  );
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
	return (1.0 / (4.0*M_PI*v)) * (
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

