#include <math.h>
#include <stdlib.h>
#include <stdio.h>
using namespace std;

#include "gamma_ab_v.h"

#ifndef M_PI
/* M_PI no longer defined in the standards */
#define M_PI 3.14159265358979
#endif

// JOURNAL OF GEOPHYSICAL RESEARCH, VOL. 98, NO. B6, PP. 9551-9555, 1993

double invtan(const double num, const double denom)
{
	if(denom == 0)
	{
		if(num > 0)
			return M_PI * 0.5;
		return -M_PI * 0.5;
	}
	return atan(num/denom);
}

static double sign(const double x)
{
	if(x > 0)
		return  1;
	if(x < 0)
		return -1;
	return 0;
}

static double phi(const double n, const double d)
{
	if(n == 0)
		return 0;
	if(d == 0)
		return 1000 * sign(n); //arcsinh(10^500) ~= 1000. 10^500 is like infinity
	const double p = n / d;
	return log(p + sqrt(1.0 + p*p));
}


static double g(const double x, const double y, const double z)
{
	const double xx = x*x;
	const double yy = y*y;
	const double zz = z*z;
	const double R = sqrt(xx+yy+zz);

	return 	  (x*y*z) * phi(z, sqrt(xx+yy))
			+ (y/6.0) * (3.0*zz-yy) * phi(x, sqrt(yy+zz))
			+ (x/6.0) * (3.0*zz-xx) * phi(y, sqrt(xx+zz))
			- ( z*zz/6.0) * invtan(x*y,(z*R))
			- ( z*yy/2.0) * invtan(x*z,(y*R))
			- ( z*xx/2.0) * invtan(y*z,(x*R))
			- (x*y*R/3.0);
}

static double G2(const double x, const double y, const double z)
{
	return g(x,y,z) - g(x,y,0);
}
static double G1(const double X, const double Y, const double Z, const double dx, const double dy, const double dz)
{
 	return G2(X+dx,Y,Z+dz) - G2(X+dx,Y,Z) - G2(X,Y,Z+dz) + G2(X,Y,Z);
}

static double G0(const double X, const double Y, const double Z, const double dx, const double dy, const double dz)
{
 	return G1(X,Y,Z,dx,dy,dz) - G1(X,Y-dy,Z,dx,dy,dz) - G1(X,Y,Z-dz,dx,dy,dz) + G1(X,Y-dy,Z-dz,dx,dy,dz);
}


static double f(const double x, const double y, const double z)
{
	const double xx = x*x;
	const double yy = y*y;
	const double zz = z*z;
	const double R = sqrt(xx+yy+zz);
	
	return 	  y * 0.5 * (zz-xx) * phi(y,sqrt(xx+zz))
			+ z * 0.5 * (yy-xx) * phi(z,sqrt(xx+yy))
			- x*y*z * invtan(y*z, x*R)
			+ (1.0/6.0) * (2.0*xx-yy-zz) * R;
}

static double F2(const double X, const double Y, const double Z)
{
 	return f(X,Y,Z) - f(X,0,Z) - f(X,Y,0) + f(X,0,0);
}

static double F1(const double X, const double Y, const double Z, const double dx, const double dy, const double dz)
{
 	return F2(X, Y, Z) - F2(X, Y-dy, Z) - F2(X, Y, Z-dz) + F2(X, Y-dy, Z-dz);
}

static double F0(const double X, const double Y, const double Z, const double dx, const double dy, const double dz)
{
 	return 	  F1(X, Y+dy,Z+dz,dx,dy,dz) 
			- F1(X, Y,   Z+dz,dx,dy,dz) 
			- F1(X, Y+dy,Z,   dx,dy,dz) 
			+ F1(X, Y,   Z,   dx,dy,dz);
}


static double Nxx(const double X, const double Y, const double Z, const double dx, const double dy, const double dz)
{
	const double tau = dx*dy*dz;
 	return -1.0 * (1.0 / (4.0*M_PI*tau)) * (2.0*F0(X,Y,Z,dx,dy,dz) - F0(X+dx, Y,Z,dx,dy,dz) - F0(X-dx,Y,Z,dx,dy,dz));
}

static double Nxy(const double X, const double Y, const double Z, const double dx, const double dy, const double dz)
{
	const double tau = dx*dy*dz;
	
 	return -1.0 * (1.0 / (4.0*M_PI*tau)) * (G0(X,Y,Z,dx,dy,dz) - G0(X-dx,Y,Z,dx,dy,dz) - G0(X,Y+dy,Z,dx,dy,dz) + G0(X-dx,Y+dy,Z,dx,dy,dz));
}

static double Nyy(const double X, const double Y, const double Z, const double dx, const double dy, const double dz)
{
	return Nxx(Y,X,Z,dy,dx,dz);
}

static double Nzz(const double X, const double Y, const double Z, const double dx, const double dy, const double dz)
{
	return Nxx(Z,Y,X,dz,dy,dx);
}

static double Nxz(const double X, const double Y, const double Z, const double dx, const double dy, const double dz)
{
	return Nxy(X,Z,Y,dx,dz,dy);
}

static double Nyz(const double X, const double Y, const double Z, const double dx, const double dy, const double dz)
{
	return Nxy(Y,Z,X,dy,dz,dx);
}


double gamma_xx_v(const double x, const double y, const double z, const double* prism)
{
	return Nxx(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_yy_v(const double x, const double y, const double z, const double* prism)
{
	return Nyy(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_zz_v(const double x, const double y, const double z, const double* prism)
{
	return Nzz(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_xy_v(const double x, const double y, const double z, const double* prism)
{
	return Nxy(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_yz_v(const double x, const double y, const double z, const double* prism)
{
	return Nyz(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_xz_v(const double x, const double y, const double z, const double* prism)
{
	return Nxz(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_yx_v(const double x, const double y, const double z, const double* prism)
{
	return Nxy(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_zy_v(const double x, const double y, const double z, const double* prism)
{
	return Nyz(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_zx_v(const double x, const double y, const double z, const double* prism)
{
	return Nxz(x,y,z,prism[0],prism[1],prism[2]);
}


