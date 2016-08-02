#include "pointfunc_demag.h"
#include <math.h>
#include <stdio.h>


#ifndef M_PI
#define M_PI 3.1415926535897932385
#endif


// IEEE TRANSACTIONS  ON MAGNETICS,  VOL.  34, NO.  1, JANUARY  1998 
 
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


static DOUBLE _atan(const DOUBLE num, const DOUBLE denom)
{
	if(denom == 0)
	{
		if(num > 0)
			return M_PI * 0.5;
		return -M_PI * 0.5;
	}
	const DOUBLE v = ATAN(num/denom);
	return v;
}

static DOUBLE powNegOne(const int e)
{
	if(e & 0x1)
		return -1;
	return 1;
}
static double Kxx(const double* x, const double* y, const double* z, const double X, const double Y, const double Z)
{
	DOUBLE sum = 0;
	for(int i=0; i<2; i++)
	{
		const DOUBLE dx = X - x[i];
		for(int j=0; j<2; j++)
		{
			const DOUBLE dy = Y - y[j];
			for(int k=0; k<2; k++)
			{
				const DOUBLE dz = Z - z[k];
				const DOUBLE D = SQRT(dx*dx + dy*dy + dz*dz);
				
				sum += powNegOne(i+j+k) * _atan(dy*dz, dx*D);
			}
		}
	}
	return sum;
}

static double Kxy(const double* x, const double* y, const double* z, const double X, const double Y, const double Z)
{
	DOUBLE sum = 0;
	for(int i=0; i<2; i++)
	{
		const DOUBLE dx = X - x[i];
		for(int j=0; j<2; j++)
		{
			const DOUBLE dy = Y - y[j];
			for(int k=0; k<2; k++)
			{
				const DOUBLE dz = Z - z[k];
				const DOUBLE D = SQRT(dx*dx+dy*dy+dz*dz);
				
				sum += powNegOne(i+j+k) *  LOG(D + dz);
			}
		}
	}
	return -1.0 * sum;
}

static void shuffle(double* dest, const double* src, const int* o)
{
	dest[0] = src[o[0]];
	dest[1] = src[o[1]];
	dest[2] = src[o[2]];
}
 
double magnetostatic_Pxx(const double X,   const double Y,  const double Z, const double* source)
{
	double x[2] = {0,0};
	double y[2] = {0,0};
	double z[2] = {0,0};
	
	x[1] = source[0];
	y[1] = source[1];
	z[1] = source[2];
	
	return Kxx(x,y,z, X,Y,Z);
}

double magnetostatic_Pxy(const double X,   const double Y,   const double Z, const double* source)
{
	double x[2] = {0,0};
	double y[2] = {0,0};
	double z[2] = {0,0};
	
	x[1] = source[0];
	y[1] = source[1];
	z[1] = source[2];
	
	return Kxy(x,y,z, X,Y,Z);	
}

double magnetostatic_Pxz(const double X,   const double Y,   const double Z, const double* source)
{
	double s[3];
	const int o[3] = {0,2,1};
	shuffle(s, source, o);
	return magnetostatic_Pxy(X, Z, Y, s);
}

double magnetostatic_Pyx(const double X,   const double Y,   const double Z, const double* source)
{
	double s[3];
	const int o[3] = {1,0,2};
	shuffle(s, source, o);
	return magnetostatic_Pxy(Y, X, Z, s);
}

double magnetostatic_Pyy(const double X,   const double Y,   const double Z, const double* source)
{
	double s[3];
	const int o[3] = {1,2,0};
	shuffle(s, source, o);
	return magnetostatic_Pxx(Y, Z, X, s);	
}

double magnetostatic_Pyz(const double X,   const double Y,   const double Z, const double* source)
{
	double s[3];
	const int o[3] = {1,2,0};
	shuffle(s, source, o);
	return magnetostatic_Pxy(Y, Z, X, s);
}


double magnetostatic_Pzx(const double X,   const double Y,   const double Z, const double* source)
{
	double s[3];
	const int o[3] = {2,0,1};
	shuffle(s, source, o);
	return magnetostatic_Pxy(Z, X, Y, s);
}

double magnetostatic_Pzy(const double X,   const double Y,   const double Z, const double* source)
{
	double s[3];
	const int o[3] = {2,1,0};
	shuffle(s, source, o);
	return magnetostatic_Pxy(Z, Y, X, s);
}

double magnetostatic_Pzz(const double X,   const double Y,   const double Z, const double* source)
{
	double s[3];
	const int o[3] = {2,0,1};
	shuffle(s, source, o);

	return magnetostatic_Pxx(Z, X, Y, s);
}

