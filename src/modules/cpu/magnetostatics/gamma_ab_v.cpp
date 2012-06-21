#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
using namespace std;

#include "gamma_ab_v.h"

#ifndef M_PI
/* M_PI no longer defined in the standards */
#define M_PI 3.14159265358979
#endif

// the vector and sorting stuff is because I wasn't sure if
// there was a difference in magnitude of the terms being 
// summed so I decided to to the old sum from smallest
// to biggest trick all the greybeards always talk about
// from back in the day when a floating point variable
// was 4 bits long. It makes no difference, I've never encountered
// a case where this trick has actually worked outside
// some specially crafted assignment question back in undergrad.
// 
// I should really get a tattoo reminding myself to not bother
// 
// Oh yeah, the important stuff, basing the demag tensor off of equations in:
// 
// JOURNAL OF GEOPHYSICAL RESEARCH, VOL. 98, NO. B6, PP. 9551-9555, 1993
// 
double invtan(double num, double denom)
{
	if(denom == 0)
	{
		if(num > 0)
			return M_PI * 0.5;
		return -M_PI * 0.5;
	}
	return atan(num/denom);
}

static double phi(double x)
{
	return log(x+sqrt(1.0+x*x));
}


static void g(vector<double>& terms, double scale, double x, double y, double z)
{
	const double R = sqrt(x*x+y*y+z*z);

	if(x*x+y*y != 0)
		terms.push_back(scale*(x*y*z) * phi(z/sqrt(x*x+y*y)));
	if(y*y+z*z != 0)
		terms.push_back(scale*(y/6.0) * (3.0*z*z-y*y) * phi(x/sqrt(y*y+z*z)));
	if(x*x+z*z != 0)
		terms.push_back(scale*(x/6.0) * (3.0*z*z-x*x) * phi(y/sqrt(x*x+z*z)));
		
	terms.push_back(-(z*z*z/6.0) * invtan(x*y,(z*R))*scale);
	terms.push_back(-(z*y*y/2.0) * invtan(x*z,(y*R))*scale);
	terms.push_back(-(z*x*x/2.0) * invtan(y*z,(x*R))*scale);
	terms.push_back(-(x*y*R/3.0) *scale);
	
// 	return sum;
}

static void G2(vector<double>& terms, const double scale, double x, double y, double z)
{
// 	return g(terms, scale*1.0, x,y,z) - g(terms, x,y,0);
	g(terms,  scale, x,y,z);
	g(terms, -scale, x,y,0);
}
static void G1(vector<double>& terms, const double scale, double X, double Y, double Z, double dx, double dy, double dz)
{
// 	return G2(X+dx,Y,Z+dz) - G2(X+dx,Y,Z) - G2(X,Y,Z+dz) + G2(X,Y,Z);
	G2(terms, scale, X+dx,Y,Z+dz);
	G2(terms,-scale, X+dx,Y,Z);
	G2(terms,-scale, X,Y,Z+dz);
	G2(terms, scale, X,Y,Z);
}

static void G(vector<double>& terms, const double scale, double X, double Y, double Z, double dx, double dy, double dz)
{
// 	return G1(X,Y,Z,dx,dy,dz) - G1(X,Y-dy,Z,dx,dy,dz) - G1(X,Y,Z-dz,dx,dy,dz) + G1(X,Y-dy,Z-dz,dx,dy,dz);
	G1(terms, scale, X,Y,Z,dx,dy,dz);
	G1(terms,-scale, X,Y-dy,Z,dx,dy,dz);
	G1(terms,-scale, X,Y,Z-dz,dx,dy,dz);
	G1(terms, scale, X,Y-dy,Z-dz,dx,dy,dz);
}


static void f(vector<double>& terms, const double scale, double x, double y, double z)
{
	const double R = sqrt(x*x+y*y+z*z);
// 	double sum = 0;
	
	if(y != 0 && (x*x+z*z) != 0)
		terms.push_back((y*0.5) * (z*z-x*x) * phi(y/sqrt(x*x+z*z)) * scale);
// 		sum += (y*0.5) * (z*z-x*x) * phi(y/sqrt(x*x+z*z));
	if(z != 0 && (x*x+y*y) != 0)
		terms.push_back((z*0.5) * (y*y-x*x) * phi(z/sqrt(x*x+y*y)) * scale);
// 		sum += (z*0.5) * (y*y-x*x) * phi(z/sqrt(x*x+y*y));

		
// 	sum -= x*y*z*atan2(y*z,x*R);
// 	sum += (1.0/6.0) * (2.0*x*x-y*y-z*z) * R;
	terms.push_back(-x*y*z*atan2(y*z,x*R) * scale);
	terms.push_back((1.0/6.0) * (2.0*x*x-y*y-z*z) * R * scale);
// 	return sum;
}



static void F2(vector<double>& terms, const double scale, double X, double Y, double Z)
{
// 	return f(X,Y,Z) - f(X,0,Z) - f(X,Y,0) + f(X,0,0);
	f(terms, scale,X,Y,Z);
	f(terms,-scale,X,0,Z);
	f(terms,-scale,X,Y,0);
	f(terms, scale,X,0,0);
}

static void F1(vector<double>& terms, const double scale, double X, double Y, double Z, double dx, double dy, double dz)
{
// 	return F2(X, Y-dy, Z-dz) - F2(X, Y, Z-dz) - F2(X,Y-dy,Z) + F2(X,Y,Z);
	F2(terms, scale, X, Y-dy, Z-dz);
	F2(terms,-scale, X, Y,    Z-dz);
	F2(terms,-scale, X, Y-dy, Z);
	F2(terms, scale, X, Y,    Z);
}

static void F(vector<double>& terms, const double scale, double X, double Y, double Z, double dx, double dy, double dz)
{
// 	return F1(X, Y+dy, Z+dz,dx,dy,dz) - F1(X, Y, Z+dz,dx,dy,dz) - F1(X,Y+dy,Z,dx,dy,dz) + F1(X,Y,Z,dx,dy,dz);
	F1(terms, scale,X, Y+dy, Z+dz,dx,dy,dz);
	F1(terms,-scale,X, Y,    Z+dz,dx,dy,dz);
	F1(terms,-scale,X, Y+dy, Z,   dx,dy,dz);
	F1(terms, scale,X, Y,    Z,   dx,dy,dz);
}


static bool myfunction (double i,double j) { return (fabs(i)<fabs(j)); }
static double smart_sum(vector<double>& t)
{
	sort(t.begin(), t.end(), myfunction);
	double sum = 0;
	for(unsigned int i=0; i<t.size(); i++)
	{
		sum += t[i];
	}
	return sum;
}

static double Nxx(double X, double Y, double Z, double dx, double dy, double dz)
{
	const double tau = dx*dy*dz;
	vector<double> terms;
	
// 	return -(1.0 / (4.0*M_PI*tau)) * (2.0*F(X,Y,Z,dx,dy,dz) - F(X+dx, Y,Z,dx,dy,dz) - F(X-dx,Y,Z,dx,dy,dz));
	F(terms,  2.0, X,Y,Z,dx,dy,dz);
	F(terms, -1.0, X+dx, Y,Z,dx,dy,dz);
	F(terms, -1.0, X-dx,Y,Z,dx,dy,dz);
	return -(1.0 / (4.0*M_PI*tau)) * smart_sum(terms);
}

static double Nxy(double X, double Y, double Z, double dx, double dy, double dz)
{
	const double tau = dx*dy*dz;
	
// 	return -(1.0 / (4.0*M_PI*tau)) * (G(X,Y,Z,dx,dy,dz) - G(X-dx,Y,Z,dx,dy,dz) - G(X,Y+dy,Z,dx,dy,dz) + G(X-dx,Y+dy,Z,dx,dy,dz));
	
	vector<double> terms;
	G(terms,  1.0, X,Y,Z,dx,dy,dz);
	G(terms, -1.0, X-dx,Y,Z,dx,dy,dz);
	G(terms, -1.0, X,Y+dy,Z,dx,dy,dz);
	G(terms,  1.0, X-dx,Y+dy,Z,dx,dy,dz);
	return -(1.0 / (4.0*M_PI*tau)) * smart_sum(terms);
}

static double Nyy(double X, double Y, double Z, double dx, double dy, double dz)
{
	return Nxx(Y,X,Z,dy,dx,dz);
}

static double Nzz(double X, double Y, double Z, double dx, double dy, double dz)
{
	return Nxx(Z,Y,X,dz,dy,dx);
}

static double Nxz(double X, double Y, double Z, double dx, double dy, double dz)
{
	return Nxy(X,Z,Y,dx,dz,dy);
}

static double Nyz(double X, double Y, double Z, double dx, double dy, double dz)
{
	return Nxy(Y,Z,X,dy,dz,dx);
}


double gamma_xx_v(double x, double y, double z, const double* prism)
{
	return Nxx(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_yy_v(double x, double y, double z, const double* prism)
{
	return Nyy(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_zz_v(double x, double y, double z, const double* prism)
{
	return Nzz(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_xy_v(double x, double y, double z, const double* prism)
{
	return Nxy(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_yz_v(double x, double y, double z, const double* prism)
{
	return Nyz(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_xz_v(double x, double y, double z, const double* prism)
{
	return Nxz(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_yx_v(double x, double y, double z, const double* prism)
{
	return Nxy(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_zy_v(double x, double y, double z, const double* prism)
{
	return Nyz(x,y,z,prism[0],prism[1],prism[2]);
}

double gamma_zx_v(double x, double y, double z, const double* prism)
{
	return Nxz(x,y,z,prism[0],prism[1],prism[2]);
}


