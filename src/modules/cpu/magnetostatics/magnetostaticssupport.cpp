/******************************************************************************
* Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#include "luacommon.h"
#include "magnetostaticssupport.h"
#include "gamma_ab_v.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "extrapolate.h"
#include <vector>
using namespace std;

#ifdef WIN32
 #include <windows.h>
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)
 #pragma warning(disable: 4996)
 #define snprintf _snprintf
#endif

#ifndef M_PI
#define M_PI 3.14159265358979
#endif
/*#define DipMagConversion (1.0 / (M_PI * 4.0))*/
#define DipMagConversion (1.0)

#include "../dipole/dipolesupport.h" //to compare for crossover

static double F1(double X, double Y, double Z)
{
	const double D = sqrt(X*X + Y*Y + Z*Z);
	
	const double t1 = X*Y*Z*atan2(Y*Z, X*D);
	      double t2 = 0.5 * Y * (Z*Z - X*X); //still needs log term 
	      double t3 = 0.5 * Z * (Y*Y - X*X); //still needs log term 
	const double t4 = (1.0/6.0) * (Y*Y + Z*Z - 2*X*X) * D;
	
	if(t2 != 0)	t2 *= log(fabs(D-Y));
	if(t3 != 0)	t3 *= log(fabs(D-Z));
	
	return t1+t2+t3+t4;
}


static double self_term1(double a, double b, double c)
{
	const double R = sqrt(a*a+b*b+c*c);
	const double Ra = sqrt(b*b+c*c);
	const double Rb = sqrt(a*a+c*c);
	const double Rc = sqrt(a*a+b*b);

	const double t1 = 6.0 * (a*a*a + b*b*b) - (4.0 * c*c*c)/(a*b);
	const double t2 = 2.0 * (a*a+b*b-2.0*c*c) * R / (a*b*c);
	const double t3 = 6.0*c*(Rb+Ra) /(a*b);
	const double t4 =-2.0 * (pow(a*a+b*b,1.5) + pow(a*a+c*c,1.5) + pow(b*b+c*c,1.5)) / (a*b*c);
	const double t5 = 12.0 * atan2(a*b, c*R);
	
	const double t6 = 3.0*a*log( (a*a+2.0*b*(b+Rc)) / (a*a) ) / c;
	const double t7 = 3.0*b*log( (b*b+2.0*a*(a+Rc)) / (b*b) ) / c;
	const double t8 = 3.0*c*log( (c*c+2.0*a*(a-Rb)) / (c*c) ) / b;
	const double t9 = 3.0*c*log( (c*c+2.0*b*(b-Ra)) / (c*c) ) / a;
	
	const double t10 = 3.0 * (b-c) * (b+c) * log( (R-a)/(R+a) ) / (b*c);
	const double t11 = 3.0 * (a-c) * (a+c) * log( (R-b)/(R+b) ) / (a*c);
	
	
	return (2*M_PI * -4.0) * (1.0/6.0) * (t1+t2+t3+t4+t5+t6+t7+t8+t9+t10+t11);
}

double self_term(double a, double b, double c)
{
  double R = sqrt(a*a + b*b + c*c);
  double R1 = sqrt(a*a + b*b);
  double R2 = sqrt(b*b + c*c);
  double R3 = sqrt(a*a + c*c);

  double r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11;
  r1  = (b*b-c*c)/(2.0*b*c)*log((R-a)/(R+a));
  r2  = (a*a-c*c)/(2.0*a*c)*log((R-b)/(R+b));
  r3  = b/(2.0*c)*log((R1+a)/(R1-a));
  r4  = a/(2.0*c)*log((R1+b)/(R1-b));
  r5  = c/(2.0*a)*log((R2-b)/(R2+b));
  r6  = c/(2.0*b)*log((R3-a)/(R3+a));
  r7  = 2.0*atan2(a*b,c*R);
  r8  = (pow(a,3.0)+pow(b,3.0)-2.0*pow(c,3.0))/(3.0*a*b*c);
  r9  = R*(a*a+b*b-2*c*c)/(3.0*a*b*c);
  r10 = (c/(a*b))*(R3+R2);
  r11 = -(pow(R1,3.0)+pow(R2,3.0)+pow(R3,3.0))/(3.0*a*b*c);
  double result = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11;
  result *= -8.0; //to make explicit the SI factor
  return result;

}



static double F2(double Z, double X, double Y)
{
	const double D = sqrt(X*X + Y*Y + Z*Z);
	
	double t1 = 0;
	double t2 = 0;
	double t3 = 0;
	
	if(D+Z != 0) t1 = -1.0 * X * Y * Z * log(fabs(D+Z));
	if(D+X != 0) t2 = (1.0/6.0) * Y * (Y*Y - 3.0*Z*Z) * log(fabs(D+X));
	if(D+Y != 0) t3 = (1.0/6.0) * X * (X*X - 3.0*Z*Z) * log(fabs(D+Y));
	
	const double t4 = 0.5*X*X*Z*atan2(Y*Z, X*D);
	const double t5 = 0.5*Y*Y*Z*atan2(X*Z, Y*D);
	const double t6 = (1.0/6.0)*Z*Z*Z*atan2(X*Y, Z*D);
	const double t7 = X*Y*D / 3.0;
	
	return t1+t2+t3+t4+t5+t6+t7;
}


static double MagInt(
	double   x, double   y, double   z,
	double ddx, double ddy, double ddz, 
	int a, int b)
{
	double ax[4];
	double ay[4];
	double az[4];
	double sn[4];
	
	ax[1] = -ddx; ax[2] = 0; ax[3] = ddx;
	ay[1] = -ddy; ay[2] = 0; ay[3] = ddy;
	az[1] = -ddz; az[2] = 0; az[3] = ddz;
	sn[1] = 1;    sn[2] = 2; sn[3] = 1;
	
#define for_ijk for(int i=1; i<=3; i++) for(int j=1; j<=3; j++) for(int k=1; k<=3; k++)
#define X (x + ax[i])
#define Y (y + ay[j])
#define Z (z + az[k])
	
	double K = 0;
	if(a == 0 && b == 0)
	{
		for_ijk	K += pow(-1.0, i+j+k-1) * sn[i] * sn[j] * sn[k] * F1(X, Y, Z);
	}
	if(a == 1 && b == 1)
	{
		for_ijk	K += pow(-1.0, i+j+k-1) * sn[i] * sn[j] * sn[k] * F1(Y, Z, X);
	}
	if(a == 2 && b == 2)
	{
		for_ijk	K += pow(-1.0, i+j+k-1) * sn[i] * sn[j] * sn[k] * F1(Z, X, Y);
	}
	
	if(a == 0 && b == 1)
	{
		for_ijk	K += pow(-1.0, i+j+k-1) * sn[i] * sn[j] * sn[k] * F2(Z, X, Y);
	}
	if(a == 0 && b == 2)
	{
		for_ijk	K += pow(-1.0, i+j+k-1) * sn[i] * sn[j] * sn[k] * F2(Y, Z, X);
	}
	if(a == 1 && b == 2)
	{
		for_ijk	K += pow(-1.0, i+j+k-1) * sn[i] * sn[j] * sn[k] * F2(X, Y, Z);
	}
	
#undef X
#undef Y
#undef Z

	K /= -1.0*ddx*ddy*ddz;
	return K;
}

typedef struct mag_dip_crossover
{
	double r2;  //crossover at this distance (squared) (initially big)
	double tol; //crossover at this tolerance (reset r)
} mag_dip_crossover;

bool equal_tol(double a, double b, double tolerance)
{
	if(a == 0 && b == 0)
		return true;
	
	if(b == 0 || a == 0)
		return false;
	
	double t = fabs( (a-b)/b );
// 	printf("%E, %e - %f, %f\n", a, b, t, tolerance);
	return t < tolerance;
}


static void getGAB_range(
	const double* ABC, 
	const double* prism,
	const int nA, const int nB, const int nC,  //width, depth, layers 
	const int ix, const int iy, const int iz,
	const int ax, const int ay, const int bx, const int by,	      
	              const int truemax, 
	double& gXX, double& gXY, double& gXZ,
	double& gYY, double& gYZ, double& gZZ,
	mag_dip_crossover& crossover)
{
	
	const double volumeP = pow(prism[0] * prism[1] * prism[2], 1.0);
	
	const double d1 = prism[0];
	const double l1 = prism[1];
	const double w1 = prism[2];
	
	const double d2 = prism[0];
	const double l2 = prism[1];
	const double w2 = prism[2];

	double magXX, magXY, magXZ;
	double dipXX, dipXY, dipXZ;
	
	double magYY, magYZ;
	double dipYY, dipYZ;
	
	double magZZ;
	double dipZZ;
	
	const int zz = iz;
	if(abs(zz) <= truemax)
    {
	for(int x=ax; x<=bx; x++)
	{
	    const int xx = x*nA+ix;
	    if(abs(xx) <= truemax)
	    {
		for(int y=ay; y<=by; y++)
		{
		    const int yy = y*nB+iy;
		    if(abs(yy) <= truemax)
		    {
				const double rx = ((double)xx)*ABC[0] + ((double)yy)*ABC[3] + ((double)zz)*ABC[6];
				const double ry = ((double)xx)*ABC[1] + ((double)yy)*ABC[4] + ((double)zz)*ABC[7];
				const double rz = ((double)xx)*ABC[2] + ((double)yy)*ABC[5] + ((double)zz)*ABC[8];
			
				const double r2 = rx*rx + ry*ry + rz*rz;
				//if(r2 != 0)

				if(r2 >= crossover.r2)
				{
					gXX += DipMagConversion * volumeP * gamma_xx_dip(rx, ry, rz);
					gXY += DipMagConversion * volumeP * gamma_xy_dip(rx, ry, rz);
					gXZ += DipMagConversion * volumeP * gamma_xz_dip(rx, ry, rz);

					gYY += DipMagConversion * volumeP * gamma_yy_dip(rx, ry, rz);
					gYZ += DipMagConversion * volumeP * gamma_yz_dip(rx, ry, rz);
					gZZ += DipMagConversion * volumeP * gamma_zz_dip(rx, ry, rz);
				}
				else
				{
					if(xx == 0 && yy == 0 && zz == 0)
					{
						magXX = gamma_xx_sv(d1, l1, w1)*2*M_PI * -4.0;
						magYY = gamma_yy_sv(d1, l1, w1)*2*M_PI * -4.0;
						magZZ = gamma_zz_sv(d1, l1, w1)*2*M_PI * -4.0;
						
						const double a = d1;
						const double b = l1;
						const double c = w1;

// 						printf("XX  %g\n", magXX);
// 						printf("YY  %g\n", magYY);
// 						printf("ZZ  %g\n\n", magZZ);
						
// 						printf("BCA %g\n", self_term(b,c,a));
// 						printf("CAB %g\n", self_term(c,a,b));
// 						printf("ABC %g\n", self_term(a,b,c));

// 						magXX = self_term(b,c,a);
// 						magYY = self_term(c,a,b);
// 						magZZ = self_term(a,b,c);
						
// 						printf("CBA %g\n", self_term(c,b,a));
// 						printf("BAC %g\n", self_term(b,a,c));
// 						printf("ACB %g\n", self_term(a,c,b));

					}
					else
					{
						magXX = gamma_xx_v(rx, ry, rz, d1, l1, w1, d2, l2, w2);
						magYY = gamma_yy_v(rx, ry, rz, d1, l1, w1, d2, l2, w2);
						magZZ = gamma_zz_v(rx, ry, rz, d1, l1, w1, d2, l2, w2);
						
// 						double XXTest = MagInt(rx, ry, rz, d1, l1, w1, 0, 0);
						
// 						magXX = MagInt(rx, ry, rz, d1, l1, w1, 0, 0);
// 						magYY = MagInt(rx, ry, rz, d1, l1, w1, 1, 1);
// 						magZZ = MagInt(rx, ry, rz, d1, l1, w1, 2, 2);
// 	double   x, double   y, double   z,
// 	double ddx, double ddy, double ddz, 
// 	int a, int b)
					}

// 					magXY = MagInt(rx, ry, rz, d1, l1, w1, 0, 1);
// 					magXZ = MagInt(rx, ry, rz, d1, l1, w1, 0, 2);
// 					magYZ = MagInt(rx, ry, rz, d1, l1, w1, 1, 2);

					magXY = gamma_xy_v(rx, ry, rz, d1, l1, w1, d2, l2, w2);
					magXZ = gamma_xz_v(rx, ry, rz, d1, l1, w1, d2, l2, w2);
					magYZ = gamma_yz_v(rx, ry, rz, d1, l1, w1, d2, l2, w2);

					dipXX = DipMagConversion * volumeP * gamma_xx_dip(rx, ry, rz);
					dipXY = DipMagConversion * volumeP * gamma_xy_dip(rx, ry, rz);
					dipXZ = DipMagConversion * volumeP * gamma_xz_dip(rx, ry, rz);

					dipYY = DipMagConversion * volumeP * gamma_yy_dip(rx, ry, rz);
					dipYZ = DipMagConversion * volumeP * gamma_yz_dip(rx, ry, rz);
					dipZZ = DipMagConversion * volumeP * gamma_zz_dip(rx, ry, rz);

					// 				if( sqrt(rx*rx+ry*ry+rz+rz) > 40)
					// 					printf("md: %E, %E\n", magXX, dipXX);

					bool same = equal_tol(magXX, dipXX, crossover.tol) &&
						equal_tol(magXY, dipXY, crossover.tol) &&
						equal_tol(magXZ, dipXZ, crossover.tol) &&
						equal_tol(magYY, dipYY, crossover.tol) &&
						equal_tol(magYZ, dipYZ, crossover.tol) &&
						equal_tol(magZZ, dipZZ, crossover.tol);

					if(same && r2 > 0)
					{
						const double a = fabs(r2);
						const double b = crossover.r2;
						
						if(a < b)
						{
							crossover.r2 = a;
							printf("crossover at %g\n", a);
						}
						else
							crossover.r2 = b;
						
						//printf("crossover at: r = %f\n", sqrt(crossover.r2));
					}

					gXX += magXX;
					gXY += magXY;
					gXZ += magXZ;

					gYY += magYY;
					gYZ += magYZ;
					gZZ += magZZ;	
				}
		    }
		}
	    }
	}
    }
}


///periodic XY
static void getGAB(
	const double* ABC, 
	const double* prism, /* 3 vector */
	const int nA, const int nB, const int nC,  //width, depth, layers 
	const int ix, const int iy, const int iz, 
	const int smin, const int smax, const int truemax, //allowing rings 
	double* XX, double* XY, double* XZ,
	double* YY, double* YZ, double* ZZ,
	mag_dip_crossover& crossover)
{
	double gXX = 0;
	double gXY = 0;
	double gXZ = 0;
	double gYY = 0;
	double gYZ = 0;
	double gZZ = 0;

	const int minx = smin;
	const int miny = smin;
	const int maxx = smax;
	const int maxy = smax;

	// +----b
	// |    |
	// |    |
	// a----+

	// each coordinate here denotes a lattice
	int ax[9];
	int ay[9];
	int bx[9];
	int by[9];

	ax[0] = -(maxx-1);
	ay[0] = miny;
	bx[0] = -minx;
	by[0] = (maxy-1);

	ax[1] = -minx+1;
	ay[1] = ay[0];
	bx[1] = minx-1;
	by[1] = by[0];

	ax[2] = minx;
	ay[2] = ay[0];
	bx[2] = maxx-1;
	by[2] = by[0];

	ax[3] = ax[0];
	ay[3] = -(miny-1);
	bx[3] = bx[0];
	by[3] = miny-1;

	ax[4] = ax[2];
	ay[4] = ay[3];
	bx[4] = bx[2];
	by[4] = by[3];

	ax[5] = ax[0];
	ay[5] = -(maxy-1);
	bx[5] = bx[0];
	by[5] = -miny;

	ax[6] = ax[1];
	ay[6] = ay[5];
	bx[6] = bx[1];
	by[6] = by[5];

	ax[7] = ax[2];
	ay[7] = ay[5];
	bx[7] = bx[2];
	by[7] = by[5];

	ax[8] = -(maxx-1);
	ay[8] = -(maxy-1);
	bx[8] = -ax[8];
	by[8] = -ay[8];

	if(smin == 0)
	{
	    const int i = 8;
// static void getGAB_range(
// 	const double* ABC, 
// 	const double* prism,
// 	const int nA, const int nB, const int nC,  //width, depth, layers 
// 	const int ix, const int iy, const int iz,
// 	const int ax, const int ay, const int bx, const int by,	      
// 	              const int truemax, 
// 	double& gXX, double& gXY, double& gXZ,
// 	double& gYY, double& gYZ, double& gZZ,
// 	mag_dip_crossover& crossover)

		getGAB_range(ABC, prism, nA, nB, nC, ix, iy, iz,
			 ax[i], ay[i], bx[i], by[i],
			 truemax, 
			 gXX, gXY, gXZ,
			 gYY, gYZ, gZZ,
			 crossover);
	}
	else
	{
	    for(int i=0; i<8; i++)
	    {
		getGAB_range(ABC, prism, nA, nB, nC, ix, iy, iz,
			     ax[i], ay[i], bx[i], by[i],
			     truemax,
			     gXX, gXY, gXZ,
			     gYY, gYZ, gZZ,
				 crossover);
	    }
	}

#ifndef WIN32
#warning This is a hack to fix self terms. Eventually this will be in the numerical code.
#endif
	if(ix == 0 && iy == 0 && iz == 0)
	{
		gXX *= 0.5;
		gXY *= 0.5;
		gXZ *= 0.5;

		gYY *= 0.5;
		gYZ *= 0.5;

		gZZ *= 0.5;
	}

	*XX = gXX;
	*XY = gXY;
	*XZ = gXZ;

	*YY = gYY;
	*YZ = gYZ;

	*ZZ = gZZ;
}//end function getGAB


static int _isZeroMat(const double* M, int nx, int ny)
{
	for(int j=0; j<ny; j++)
	{
		for(int i=0; i<nx; i++)
		{
			if(fabs(M[i+j*nx]) > 1E-16)
				return 0;
		}
	}
	return 1;
}

static void _writemat(FILE* f, const char* name, int zoffset, const double* M, int nx, int ny)
{
	fprintf(f, "\n");
	
	if(_isZeroMat(M, nx, ny))
	{
		fprintf(f, "%s[%i] = 0\n", name, zoffset);
		return;
	}
	
	fprintf(f, "%s[%i] = [[\n", name, zoffset);

	for(int j=0; j<ny; j++)
	{
// 		fprintf(f, "    {");
		for(int i=0; i<nx; i++)
		{
// 			fprintf(f, "% 12e%s", M[i+j*nx], i==(nx-1)?"}":", ");
			fprintf(f, "% 12e%s", M[i+j*nx], (i==(nx-1) && j==(ny-1))?"":",");
		}
		fprintf(f, "\n");
// 		fprintf(f, "%c\n", j==(ny-1)?' ':',');
	}
	fprintf(f, "]]\n");
	fprintf(f, "\n");
}

static void _writeParser(FILE* f)
{
	const char* parse =
		"\n"
		"function tokenizeNumbers(line)\n"
		"	local t = {}\n"
		"	for w in string.gmatch(line, \"[^,]+\") do\n"
		"		table.insert(t, tonumber(w))\n"
		"	end\n"
		"	return t\n"
		"end\n"
		"\n"
		"function tokenizeLines(lines)\n"
		"	-- strip empty lines\n"
		"	lines = string.gsub(lines, \"^%s*\\n*\", \"\")\n"
		"	lines = string.gsub(lines, \"\\n\\n+\", \"\\n\")\n"
		"	\n"
		"	local t = {}\n"
		"	for w in string.gmatch(lines, \"(.-)\\n\" ) do\n"
		"		table.insert(t, tokenizeNumbers(w))\n"
		"	end\n"
		"	\n"
		"	return t\n"
		"end\n"
		"\n"
		"function parseMatrix(M)\n"
		"	if M == 0 then\n"
		"		-- returns a 2D table that always returns zero\n"
		"		local tz, ttz = {}, {}\n"
		"		setmetatable(tz,  {__index = function() return  0 end})\n"
		"		setmetatable(ttz, {__index = function() return tz end})\n"
		"		return ttz\n"
		"	end\n"
		"	\n"
		"	return tokenizeLines(M)\n"
		"end\n"
		"\n"
		"function map(f, t)\n"
		"	for k,v in pairs(t) do\n"
		"		t[k] = f(v)\n"
		"	end\n"
		"	return t\n"
		"end\n"
		"\n"
		"function parse()\n"
		"	XX = map(parseMatrix, XX)\n"
		"	XY = map(parseMatrix, XY)\n"
		"	XZ = map(parseMatrix, XZ)\n"
		"\n"
		"	YY = map(parseMatrix, YY)\n"
		"	YZ = map(parseMatrix, YZ)\n"
		"\n"
		"	ZZ = map(parseMatrix, ZZ)\n"
		"end\n";

	fprintf(f, "%s", parse);
}

static bool magnetostatics_write_matrix(const char* filename,
	const double* ABC,
	const double* prism,
	const int nx, const int ny, const int nz,  //width, depth, layers 
	const int gmax, 
	const double* XX, const double* XY, const double* XZ,
	const double* YY, const double* YZ, const double* ZZ)
{
	FILE* f = fopen(filename, "w");
	if(!f)
		return false;
	
	fprintf(f, "-- This file contains magnetostatic interaction matrices\n");
	fprintf(f, "\n");
	if(gmax == -1)
		fprintf(f, "gmax = math.huge\n");
	else
		fprintf(f, "gmax = %i\n", gmax);
	fprintf(f, "nx, ny, nz = %i, %i, %i\n", nx, ny, nz);
	fprintf(f, "cellDimensions = {%g, %g, %g}\n", prism[0], prism[1], prism[2]);
	fprintf(f, "ABC = {{%g, %g, %g}, --unit cell\n       {%g, %g, %g},\n       {%g, %g, %g}}\n\n", 
		ABC[0], ABC[1], ABC[2],
		ABC[3], ABC[4], ABC[5],
		ABC[6], ABC[7], ABC[8]);
	fprintf(f, "XX={} XY={} XZ={} YY={} YZ={} ZZ={}\n");
	
	int c = 0;
	for(int zoffset=0; zoffset<nz; zoffset++)
	{
		_writemat(f, "XX", zoffset, &XX[c*nx*ny], nx, ny);
		_writemat(f, "XY", zoffset, &XY[c*nx*ny], nx, ny);
		_writemat(f, "XZ", zoffset, &XZ[c*nx*ny], nx, ny);
		
		_writemat(f, "YY", zoffset, &YY[c*nx*ny], nx, ny);
		_writemat(f, "YZ", zoffset, &YZ[c*nx*ny], nx, ny);
		
		_writemat(f, "ZZ", zoffset, &ZZ[c*nx*ny], nx, ny);
		
		c++;
	}
	
	_writeParser(f);
	
	fclose(f);
	return true;
}

static void next_magnetostaticsfilename(const char* current, char* next, int len, const int nx, const int ny)
{
	if(current && current[0])
	{
		int x, y, v;
		sscanf(current, "GAB_%ix%i.%i.lua", &x, &y, &v);
		snprintf(next, len, "GAB_%ix%i.%i.lua", x, y, v+1);
	}
	else
	{
		snprintf(next, len, "GAB_%ix%i.%i.lua", nx, ny, 1);
	}
}

static int file_exists(const char* filename)
{
	FILE* f = fopen(filename, "r");
	if(f)
	{
		fclose(f);
		return 1;
	}
	return 0;
}


static bool valueMatch(lua_State* L, const char* name, int value)
{
	lua_getglobal(L, name);
	if(!lua_isnumber(L, -1))
	{
		lua_pop(L, 1);
		return false;
	}
	
	int v = lua_tointeger(L, -1);
	lua_pop(L, 1);
	return v == value;
}

static bool approxSame(double a, double b)
{
	bool c = fabs(a-b) <= 0.5*(fabs(a) + fabs(b)) * 1e-6;
	return c;
}

static bool checkTable(lua_State* L, const double* v3)
{
	if(!lua_istable(L, -1))
	{
		return false;
	}
	for(int i=0; i<3; i++)
	{
		lua_pushinteger(L, i+1);
		lua_gettable(L, -2);
		if(!lua_isnumber(L, -1) || !(approxSame(lua_tonumber(L, -1), v3[i])))
		{
			lua_pop(L, 1);
			return false;
		}
		lua_pop(L, 1);
	}
	return true;
}

static bool magnetostaticsParamsMatch(
	const char* filename,
	const int nx, const int ny, const int nz,
	const int gmax, const double* ABC, const double* prism)
{
	lua_State *L = lua_open();
	luaL_openlibs(L);
	
	if(luaL_dofile(L, filename))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		lua_close(L);
		return false;
	}
	
	const char* nn[3] = {"nx", "ny", "nz"};
	int  nv[3]; 
	nv[0] = nx; 
	nv[1] = ny; 
	nv[2] = nz; 

	for(int i=0; i<3; i++)
	{
		if(!valueMatch(L, nn[i], nv[i]))
		{
			lua_close(L);
			return false;
		}
	}


	int file_gmax = 0;
	lua_getglobal(L, "gmax");
	file_gmax = lua_tointeger(L, -1);
	lua_getglobal(L, "math");
	lua_pushstring(L, "huge");
	lua_gettable(L, -2);
	lua_remove(L, -2); //remove math table
	if(lua_equal(L, -2, -1)) //then gmax = math.huge
	{
		file_gmax = -1; //special marker for math.huge
	}
	lua_pop(L, 2);
	
	if(file_gmax != gmax)
	{
		lua_close(L);
		return false;
	}

	lua_getglobal(L, "cellDimensions");
	if(!checkTable(L, prism))
	{
		lua_close(L);
		return false;
	}
	
	//see if unit cell matches
	lua_getglobal(L, "ABC");
	if(!lua_istable(L, -1))
	{
		lua_close(L);
		return false;
	}
	
	for(int i=0; i<3; i++)
	{
		lua_pushinteger(L, i+1);
		lua_gettable(L, -2); // get A/B/C
		if(!checkTable(L, ABC+3*i))
		{
			lua_close(L);
			return false;
		}
		lua_pop(L, 1);
	}
	lua_close(L);
	return true;
}

static void loadXYZ(
	const char* filename,
	const int nx, const int ny, const int nz,
	double* XX, double* XY, double* XZ,
	double* YY, double* YZ, double* ZZ)
{
	lua_State* L = lua_open();
	luaL_openlibs(L);
	
	const char* vars[6] = {"XX", "XY", "XZ", "YY", "YZ", "ZZ"};
	double* arrs[6];
	arrs[0] = XX;
	arrs[1] = XY;
	arrs[2] = XZ;
	arrs[3] = YY;
	arrs[4] = YZ;
	arrs[5] = ZZ;
	
	if(luaL_dofile(L, filename))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		lua_close(L);
		return;
	}

	lua_getglobal(L, "parse");
	if(lua_isfunction(L, -1))
	{
		if(lua_pcall(L, 0, 0, 0))
	    {
			fprintf(stderr, "%s\n", lua_tostring(L, -1));
			lua_close(L);
			return;	      
	    }
	}
	
	const int nxyz = nx*ny*nz;
	for(int a=0; a<6; a++)
	{
		int c = 0;
		//int p = 0;
		lua_getglobal(L, vars[a]); //XX
		for(int k=0; k<nz; k++)
		{
			lua_pushinteger(L, k); //XX 0
			lua_gettable(L, -2);   //XX XX[0]
			for(int j=0; j<ny; j++)
			{
				lua_pushinteger(L, j+1); // XX XX[0] 1
				lua_gettable(L, -2);     // XX XX[0] XX[0,1]
				for(int i=0; i<nx; i++)
				{
					lua_pushinteger(L, i+1); // XX XX[0] XX[0,1] 2
					lua_gettable(L, -2);     // XX XX[0] XX[0,1] XX[0,1,2]
					arrs[a][c*nx*ny + j*nx + i] = lua_tonumber(L, -1);
					lua_pop(L, 1); // XX XX[0] XX[0,1]
				}
				lua_pop(L, 1); // XX XX[0]
			}
			lua_pop(L, 1); // XX
			c++;
		}
		lua_pop(L, 1); //
	}
	
	lua_close(L);
}

static bool extrapolate(lua_State* L,
			vector<int>& cuttoffs,
            vector<double>& vXX, vector<double>& vXY, vector<double>& vXZ, 
                                 vector<double>& vYY, vector<double>& vYZ, 
                                                      vector<double>& vZZ, bool& fail)
{
	vector< vector<double> > vAB;
	vAB.push_back(vXX);
	vAB.push_back(vXY);
	vAB.push_back(vXZ);
	vAB.push_back(vYY);
	vAB.push_back(vYZ);
	vAB.push_back(vZZ);
	
	double sol[6];
	bool   res[6];
	bool   ok = true;
	for(int i=0; i<6; i++)
	{
		lua_pop(L, lua_gettop(L));
		lua_getglobal(L, "extrapolate");
		lua_newtable(L);
		for(unsigned int j=0; j<cuttoffs.size(); j++)
		{
			lua_pushinteger(L, j+1);
			lua_newtable(L); //{x,y} holder
			lua_pushinteger(L, 1);
			lua_pushinteger(L, cuttoffs[j]);
			lua_settable(L, -3); //set x
			lua_pushinteger(L, 2);
			lua_pushnumber(L, vAB[i][j]);
			lua_settable(L, -3); // set y
			lua_settable(L, -3); // add {x,y} pair
		}
		
		//int lua_pcall (lua_State *L, int nargs, int nresults, int errfunc);

		if(lua_pcall(L, 1, 1, 0))
		{
			fprintf(stderr, "%s\n", lua_tostring(L, -1));
			fail = true;
			return false;
		}
		
		int t = lua_type(L, -1);
		
		if(t == LUA_TBOOLEAN)
		{
			res[i] = false;
		}
		if(t == LUA_TNUMBER)
		{
			res[i] = true;
			sol[i] = lua_tonumber(L, -1);
		}
		lua_pop(L, 1);
		ok &= res[i];
	}
	
	fail = false;
	if(ok)
	{
		vXX.push_back(sol[0]);
		vXY.push_back(sol[1]);
		vXZ.push_back(sol[2]);
		vYY.push_back(sol[3]);
		vYZ.push_back(sol[4]);
		vZZ.push_back(sol[5]);
		return true;
	}
	return false;
}

void magnetostaticsLoad(
	const int nx, const int ny, const int nz,
	const int gmax, const double* ABC,
	const double* prism, /* 3 vector */
	double* XX, double* XY, double* XZ,
	double* YY, double* YZ, double* ZZ,
	double tol)
{
	char fn[1024] = ""; //mmm arbitrary
	
	lua_State* L = lua_open();
	luaL_openlibs(L);
	
	mag_dip_crossover crossover;
	crossover.r2  = 1E10;
	crossover.tol = tol;
	
	
	while(true)
	{
		next_magnetostaticsfilename(fn, fn, 64, nx, ny);
		if(file_exists(fn))
		{
			if(magnetostaticsParamsMatch(fn, nx, ny, nz, gmax, ABC, prism))
			{
				loadXYZ(fn, 
						nx, ny, nz,
						XX, XY, XZ,
						YY, YZ, ZZ);
				lua_close(L);
				return;
			}
		}
		else
		{
			int c = 0;
			for(int k=0; k<nz; k++)
			for(int j=0; j<ny; j++)
			for(int i=0; i<nx; i++)
			{
				fflush(stdout);
				if(gmax != -1)
				{
// 					printf("c = %i\n", c);
					getGAB(ABC,
							prism,
							nx, ny, nz,
							i, j, k,
							0, gmax, gmax,
							XX+c, XY+c, XZ+c,
							YY+c, YZ+c, ZZ+c, 
							crossover);
					
// 						c++;
				}
				else // math.huge sum
				{
					vector<double> vXX;
					vector<double> vXY;
					vector<double> vXZ;
					vector<double> vYY;
					vector<double> vYZ;
					vector<double> vZZ;
					vector<int> cuttoffs;
					
					double tXX, tXY, tXZ, tYY, tYZ, tZZ;
					double sXX, sXY, sXZ, sYY, sYZ, sZZ;
					sXX=0; sXY=0; sXZ=0;
					sYY=0; sYZ=0; sZZ=0;

					const int lstep = 4;
					bool fail = false;
					int _lmin = 0; //inits of lattices
					int _lmax = lstep;
					
					bool converge = false;
					int q = 0;
					int maxiter = 5000;
					do
					{
						getGAB(ABC,
							prism,
							nx, ny, nz,
							i, j, k,
							_lmin, _lmax, (int)1e16,
							 &tXX, &tXY, &tXZ, &tYY, &tYZ, &tZZ,
							crossover);
						
						sXX+=tXX;sXY+=tXY;sXZ+=tXZ;
						sYY+=tYY;sYZ+=tYZ;sZZ+=tZZ;
						vXX.push_back(sXX);
						vXY.push_back(sXY);
						vXZ.push_back(sXZ);
						vYY.push_back(sYY);
						vYZ.push_back(sYZ);
						vZZ.push_back(sZZ);

						cuttoffs.push_back(_lmax);
						_lmin += lstep;
						_lmax += lstep;
						if(q>=8) //let the system prime itself before trying to extrapolate
						{
// 						    converge = true;
							converge = extrapolate(L, cuttoffs, vXX, vXY, vXZ, vYY, vYZ, vZZ, fail);
							if(converge)
							{
								printf("Extrapolating at %i\n", _lmax);
							}
						}
						q++;
						maxiter--;
					}while(!converge && !fail && maxiter);
				
// 					if(converge || fail)
					{
					    int last = vXX.size() - 1;
					    
					    XX[c] = vXX[last];
					    XY[c] = vXY[last];
					    XZ[c] = vXZ[last];

					    YY[c] = vYY[last];
					    YZ[c] = vYZ[last];

					    ZZ[c] = vZZ[last];
					}
	
					if(fail | !maxiter)
					{
					    fprintf(stderr, "Failed to find a extrapolate to solution under tolerance, using best calculated value\n");
					}
				}
				c++;
			}
			
			magnetostatics_write_matrix(fn,
				ABC, prism,
				nx, ny, nz,
				gmax,
				XX, XY, XZ,
				YY, YZ, ZZ);
			lua_close(L);
			return;
		}
	}
	//never reaches here
	lua_close(L);
}


