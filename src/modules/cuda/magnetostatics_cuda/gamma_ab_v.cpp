#include <math.h>
#include <stdio.h>
#include "cubature.h"
#include "gamma_ab_v.h"

#ifndef M_PI
/* M_PI no longer defined in the standards */
#define M_PI 3.14159265358979
#endif

void lnf(unsigned ndim, unsigned npts, const double *x, void *fdata, unsigned fdim, double *fval)
{
  /*Integration function for ln(f±±±). Uses fdata(sigma) to know if it's x,y or z*/
  /*double factor = 1.0/(4.0*M_PI);*/
  double factor = 1.0;
  int sigma = *((int *) fdata);
  double sqrt_term = 0.0;
  unsigned i,j;
  for (j = 0; j < npts; ++j)
  {
    for (i = 0; i < ndim; ++i)
      sqrt_term += (x[j*ndim+i]*x[j*ndim+i]);
    fval[j] = factor*log(x[j*ndim+sigma] + sqrt(sqrt_term));
    sqrt_term = 0.0;
  }
}

void arctang(unsigned ndim, unsigned npts, const double *x, void *fdata, unsigned fdim, double *fval)
{
  /*Integration function for arctan(f±±±). Uses fdata(sigma) to know if it's x,y or z*/
  /*double factor = 1.0/(4.0*M_PI);*/
  double factor = 1.0;
  int sigma = *((int *) fdata);
  int sigma_oa = (sigma+1)%3;
  int sigma_ob = (sigma+2)%3;
  double sqrt_term = 0.0;
  unsigned i,j;
  for (j = 0; j < npts; ++j)
  {
    for (i = 0; i < ndim; ++i)
      sqrt_term += (x[j*ndim+i]*x[j*ndim+i]);
    fval[j] = factor*atan2(x[j*ndim+sigma_oa]*x[j*ndim+sigma_ob],x[j*ndim+sigma]*sqrt(sqrt_term));
    sqrt_term = 0.0;
  }
}

#define _vM(a, b, c) max_##a##b##c[0] = x##a##p;\
                     max_##a##b##c[1] = y##b##p;\
                     max_##a##b##c[2] = z##c##p; 

#define _vm(a, b, c) min_##a##b##c[0] = x##a##m;\
                     min_##a##b##c[1] = y##b##m;\
                     min_##a##b##c[2] = z##c##m; 

#define _v(a, b, c) _vM(a,b,c) _vm(a,b,c)
					 
double gamma_v(const double x, const double y, const double z, 
			   const double d1, const double l1, const double w1, 
			   const double d2, const double l2, const double w2, 
			   int sigma, const int a_neq_b)
{
  double V;
  /*Function that calls the numerical integrator. Last two parameters dictate which function is used.
  sigma: 0 = x, 1 = y, 2 = z, passed to the integrand to choose x,y,z*/
  /*Integration limits*/
  const double xpp = x + (d1+d2)/2.0;
  const double xpm = x + (d1-d2)/2.0;
  const double xmp = x - (d1-d2)/2.0;
  const double xmm = x - (d1+d2)/2.0;

  const double ypp = y + (l1+l2)/2.0;
  const double ypm = y + (l1-l2)/2.0;
  const double ymp = y - (l1-l2)/2.0;
  const double ymm = y - (l1+l2)/2.0;

  const double zpp = z + (w1+w2)/2.0;
  const double zpm = z + (w1-w2)/2.0;
  const double zmp = z - (w1-w2)/2.0;
  const double zmm = z - (w1+w2)/2.0;

  /* The following produces: initializer element is not computable at load time 
  double max_mpm[3] = {xmp,ypp,zmp}, min_mpm[3] = {xmm,ypm,zmm};
  double max_mmp[3] = {xmp,ymp,zpp}, min_mmp[3] = {xmm,ymm,zpm};
  double max_ppp[3] = {xpp,ypp,zpp}, min_ppp[3] = {xpm,ypm,zpm};
  double max_pmm[3] = {xpp,ymp,zmp}, min_pmm[3] = {xpm,ymm,zmm};
  double max_mpp[3] = {xmp,ypp,zpp}, min_mpp[3] = {xmm,ypm,zpm};
  double max_mmm[3] = {xmp,ymp,zmp}, min_mmm[3] = {xmm,ymm,zmm};
  double max_ppm[3] = {xpp,ypp,zmp}, min_ppm[3] = {xpm,ypm,zmm};
  double max_pmp[3] = {xpp,ymp,zpp}, min_pmp[3] = {xpm,ymm,zpm};
  */

	double max_mpm[3], min_mpm[3];
	double max_mmp[3], min_mmp[3];
	double max_ppp[3], min_ppp[3];
	double max_pmm[3], min_pmm[3];
	double max_mpp[3], min_mpp[3];
	double max_mmm[3], min_mmm[3];
	double max_ppm[3], min_ppm[3];
	double max_pmp[3], min_pmp[3];

	/*Bookkeeping*/
	double val[8];
	double err[8];
	
	/* used in return value */
	double value = 0.0;
	unsigned i;
	
	/* This macro based method has been tested */
	/* fill out array values */
	_v(m, m, m);
	_v(m, m, p);
	_v(m, p, m);
	_v(m, p, p);
	_v(p, m, m);
	_v(p, m, p);
	_v(p, p, m);
	_v(p, p, p); 
  
  
  
#define TOL 1E-8
  if(a_neq_b)
  {
    /*Do the numerical integrals = SSSln(fmpm*fmmp*fppp*fpmm/(fmpp*fmmm*fppm*fpmp)) 
     or ln(fmpm) + ln(fmmp) + . + . - . - . - . - .*/
    adapt_integrate_v(1, lnf, &sigma, 3, min_mpm, max_mpm, 0, 0, TOL, &val[0], &err[0]);
    adapt_integrate_v(1, lnf, &sigma, 3, min_mmp, max_mmp, 0, 0, TOL, &val[1], &err[1]);
    adapt_integrate_v(1, lnf, &sigma, 3, min_ppp, max_ppp, 0, 0, TOL, &val[2], &err[2]);
    adapt_integrate_v(1, lnf, &sigma, 3, min_pmm, max_pmm, 0, 0, TOL, &val[3], &err[3]);

    adapt_integrate_v(1, lnf, &sigma, 3, min_mpp, max_mpp, 0, 0, TOL, &val[4], &err[4]);
    adapt_integrate_v(1, lnf, &sigma, 3, min_mmm, max_mmm, 0, 0, TOL, &val[5], &err[5]);
    adapt_integrate_v(1, lnf, &sigma, 3, min_ppm, max_ppm, 0, 0, TOL, &val[6], &err[6]);
    adapt_integrate_v(1, lnf, &sigma, 3, min_pmp, max_pmp, 0, 0, TOL, &val[7], &err[7]);
  }
  else
  {
    /*Do the numerical integrals = SSSarctan(gmpp)+arctan(gmmm)+arctan(gppm)+arctan(gpmp)
      -arctan(gmpm)-arctan(gmmp)-arctan(gppp)-arctan(gpmm)*/
    adapt_integrate_v(1, arctang, &sigma, 3, min_mpp, max_mpp, 0, 0, TOL, &val[0], &err[0]);
    adapt_integrate_v(1, arctang, &sigma, 3, min_mmm, max_mmm, 0, 0, TOL, &val[1], &err[1]);
    adapt_integrate_v(1, arctang, &sigma, 3, min_ppm, max_ppm, 0, 0, TOL, &val[2], &err[2]);
    adapt_integrate_v(1, arctang, &sigma, 3, min_pmp, max_pmp, 0, 0, TOL, &val[3], &err[3]);

    adapt_integrate_v(1, arctang, &sigma, 3, min_mpm, max_mpm, 0, 0, TOL, &val[4], &err[4]);
    adapt_integrate_v(1, arctang, &sigma, 3, min_mmp, max_mmp, 0, 0, TOL, &val[5], &err[5]);
    adapt_integrate_v(1, arctang, &sigma, 3, min_ppp, max_ppp, 0, 0, TOL, &val[6], &err[6]);
    adapt_integrate_v(1, arctang, &sigma, 3, min_pmm, max_pmm, 0, 0, TOL, &val[7], &err[7]);
  }

/*Gather the result*/
  for(i = 0; i < 4; ++i)
    value += val[i];
  for(i = 4; i < 8; ++i)
    value -= val[i];
  V = d2*l2*w2;
  return value/V;
}

double Gv_self(double a, double b, double c)
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
  result *= 4.0; //to make explicit the SI factor
  return result;

}

double gamma_aa(double d, double l, double w, int sigma)
{
  double ds, ls, ws;
  switch(sigma)
  {
    case 0: ws = d, ds = l, ls = w;
       break;
    case 1: ls = d, ws = l, ds = w;
       break;
    case 2: ds = d, ls = l, ws = w;
       break;
    default: ds = 0, ls = 0, ws = 0;
       break;
  }
  return Gv_self(ds,ls,ws)/(4.0*M_PI); /*SI*/ /*A.A.*/
}



#undef _vM
#undef _vm
#undef _v


/*Various gamma functions 

___
|  | ab       / xx xy xz \
|         =   | yx yy yz | 
|v   ij       \ zx zy zz /

(ab = alpha beta, aka xx, xy, yz, etc)

Every grain pair (i,j) has its own 3x3 gamma matrix.
Call with different x,y,z as the distance between their centers,
usually a multiple of d, l and w respectively.
Possibly easier to manage with double* or struct, depending on code.
 e.g., cubes (1,1,1) a few grains over

int main(int argc, char *argv[])
{
  double d1 = 1.0;
  double l1 = 1.0;
  double w1 = 1.0; 
  double d2 = 1.0;
  double l2 = 1.0;
  double w2 = 1.0; 
  double x = 1.0*d1;
  double y = 2.0*l1;
  double z = 0.0*w1;
  double value = gamma_yy_v(x,y,z,d1,l1,w1,d2,l2,d2);
  printf("Computed integral = %0.10g\n", value);
  return 0;
}

v is for volume (since the triple integral over the second cuboid is calculated therein),
not to be confused with adapt_integrate_v (vectorized, from cubature)

*/

double gamma_xx_v(double x, double y, double z, double d1, double l1, double w1, double d2, double l2, double w2)
{
  return gamma_v(x,y,z,d1,l1,w1,d2,l2,w2,0,0); /*Gx*/
}

double gamma_yy_v(double x, double y, double z, double d1, double l1, double w1, double d2, double l2, double w2)
{
  return gamma_v(x,y,z,d1,l1,w1,d2,l2,w2,1,0); /*Gy*/
}

double gamma_zz_v(double x, double y, double z, double d1, double l1, double w1, double d2, double l2, double w2)
{
  return gamma_v(x,y,z,d1,l1,w1,d2,l2,w2,2,0); /*Gz*/
}

double gamma_xy_v(double x, double y, double z, double d1, double l1, double w1, double d2, double l2, double w2)
{
  return gamma_v(x,y,z,d1,l1,w1,d2,l2,w2,2,1); /*Fz*/
}

double gamma_yx_v(double x, double y, double z, double d1, double l1, double w1, double d2, double l2, double w2)
{
  return gamma_v(x,y,z,d1,l1,w1,d2,l2,w2,2,1); /*Fz*/
}

double gamma_xz_v(double x, double y, double z, double d1, double l1, double w1, double d2, double l2, double w2)
{
  return gamma_v(x,y,z,d1,l1,w1,d2,l2,w2,1,1); /*Fy*/
}

double gamma_zx_v(double x, double y, double z, double d1, double l1, double w1, double d2, double l2, double w2)
{
  return gamma_v(x,y,z,d1,l1,w1,d2,l2,w2,1,1); /*Fy*/
}

double gamma_yz_v(double x, double y, double z, double d1, double l1, double w1, double d2, double l2, double w2)
{
  return gamma_v(x,y,z,d1,l1,w1,d2,l2,w2,0,1); /*Fx*/
}

double gamma_zy_v(double x, double y, double z, double d1, double l1, double w1, double d2, double l2, double w2)
{
  return gamma_v(x,y,z,d1,l1,w1,d2,l2,w2,0,1); /*Fx*/
}

double gamma_xx_sv(double d, double l, double w)
{
  return gamma_aa(d,l,w,0); /*like Gx*/
}

double gamma_yy_sv(double d, double l, double w)
{
  return gamma_aa(d,l,w,1); /*like Gy*/
}

double gamma_zz_sv(double d, double l, double w)
{
  return gamma_aa(d,l,w,2); /*like Gz*/
}
