#include "fmm_math.h"
#include <complex>
#include <math.h>
#include <stdio.h>
using namespace std;





#ifndef M_PI
#define M_PI 3.1415926535897932384
#endif

#ifndef SQRT2 
#define SQRT2 1.41421356237309504880
#endif

static double faci(int x)
{
	double f = 1;
	for(; x>1; x--)
		f *= x;
	return f;
}



double negOnePow(const int l)
{
	if(l < 0)
		return 1.0;

	if(l & 0x1) //odd
		return -1.0;
	return 1.0;
}

complex<double> Ylm(const int l, const int m, const double theta, const double phi)
{
	// Journal of Computational Physics 227 (2008) 1836–1862
	// "We emphasize that our spherical harmonics are considered as identically null for n < 0  or |l| > n."
	if(l < 0)
		return complex<double>(0,0);
	if(abs(m) > l)
		return complex<double>(0,0);
	//

	const double a = faci(l - abs(m));
	const double b = faci(l + abs(m));
	const double c = Plm(l, abs(m), cos(theta));

	double S, C;
	sincos(((double)m) * phi, &S, &C);

	const double d = sqrt(a/b) * c;

	return negOnePow(l) * complex<double>(d*C, d*S);
}


double Anl(int n, int l)
{
	double a = -1;
	if((n&0x1)==0) //then n is even
		a = 1;

	const double b = faci(n-l);
	const double c = faci(n+l);

	return a / sqrt(b*c);
}

complex<double> iPow(int l)
{
	l = abs(l);
	l &= 3;
	complex<double> iL;
	switch(l)
	{
	case 0: iL = complex<double>( 1, 0); break;
	case 1: iL = complex<double>( 0, 1); break;
	case 2: iL = complex<double>(-1, 0); break;
	case 3: iL = complex<double>( 0,-1); break;
	}
	return iL;
}
// Journal of Computational Physics 227 (2008) 1836–1862
complex<double> Outter(const monopole& r, int n, int l)
{
	// Journal of Computational Physics 227 (2008) 1836–1862
	// "We emphasize that our spherical harmonics are considered as identically null for n < 0  or |l| > n."
	if(n < 0)
		return complex<double>(0,0);
	if(abs(l) > n)
		return complex<double>(0,0);

	const double a = negOnePow(n);
	complex<double> iL = iPow(l);

	const double b = Anl(n,l);
	const complex<double> c = Ylm(n,l,r.t, r.p);
	const double d = pow(r.r, n+1);

//		printf("a  %g\n", a);
//		printf("iL %g %g\n", iL.real(), iL.imag());
//		printf("b  %g\n", b);
//		printf("c  %g %g\n", c.real(), c.imag());
//		printf("d  %g\n", d);

	return a*iL/b * c/d;
}


complex<double> Inner(const monopole& r, int n, int l)
{
	// Journal of Computational Physics 227 (2008) 1836–1862
	// "We emphasize that our spherical harmonics are considered as identically null for n < 0  or |l| > n."
	if(n < 0)
		return complex<double>(0,0);
	if(abs(l) > n)
		return complex<double>(0,0);


	const complex<double> a = 1.0 / iPow(l);
	const double b = Anl(n,l);
	const complex<double> c = Ylm(n,l,r.t, r.p);
	const double d = pow(r.r, n);

//	printf("a %g %g\n", a.real(), a.imag());
//	printf("b %g\n", b);
//	printf("c %g %g\n", c.real(), c.imag());
//	printf("d %g\n", d);

	return a * b * c * d;
}


void physics_spherical_coords(const double rx, const double ry, const double rz, 
							  double& theta, double& phi, double& rad)
{
	rad = rx*rx+ry*ry+rz*rz;
	
	if(rad == 0)
	{
		theta = M_PI * 0.5;
		phi = 0;
		return;
	}
	
	rad = sqrt(rad);
	if(rx == ry && rx == 0)
		phi = 0;
	else
		phi = atan2(ry, rx);
	theta = acos(rz / rad);

}
void physics_spherical_coords(const double* rxyz, double& theta, double& phi, double& rad)
{
	physics_spherical_coords(rxyz[0], rxyz[1], rxyz[2], theta, phi, rad);
}


void monopole::calcSpherical()
{
	physics_spherical_coords(x, y, z, t, p, r);
}

monopole& monopole::operator+=(const monopole& rhs)
{
	x += rhs.x;	y += rhs.y;
	z += rhs.z;	q += rhs.q;
	calcSpherical();
	return *this;
}
monopole& monopole::operator-=(const monopole& rhs)
{
	x -= rhs.x;	y -= rhs.y;
	z -= rhs.z;	q -= rhs.q;
	calcSpherical();
	return *this;
}
monopole& monopole::operator*=(const double rhs)
{
	x *= rhs;	y *= rhs;
	z *= rhs;	q *= rhs;
	calcSpherical();
	return *this;
}



int tensor_element_count(const int order)
{
	if(order < 0)
		return 0;
	return (2*order+1) + tensor_element_count(order-1);
}


static int degree_order_to_index(const int l, const int m)
{
	if(abs(m) > l)
		return -1;
	return tensor_element_count(l-1) + l + m;
}



void tensor_mat_mul(const complex<double>* A, const complex<double>* x, complex<double>* b, int max_order)
{
	int m = tensor_element_count(max_order);

	for(int r=0; r<m; r++)
	{
		b[r] = 0;
	}


	for(int r=0; r<m; r++)
	{
		for(int c=0; c<m; c++)
		{
			b[r] += A[r*m+c] * x[c];
		}
	}
}

// Journal of Computational Physics 227 (2008) 1836–1862
// Theorem 3
complex<double>* i2i_trans_mat(const int max_order, const monopole& d)
{
	int m = tensor_element_count(max_order);

	complex<double>* A = new complex<double>[m*m];

	for(int i=0; i<m*m; i++)
		A[i] = 0;

	for(int n=0; n<=max_order; n++)
	{
		for(int l=-n; l<=n; l++)
		{
			const int r = degree_order_to_index(n,l); //row
			for(int j=0; j<=max_order; j++)
			{
				for(int k=-j; k<=j; k++)
				{
					int src_degree = n-j;
					int src_order = l-k;
					const int c = degree_order_to_index(src_degree, src_order); //col
					if(c >= 0 && c < m) //then valid
					{
						double sign = 1.0;
						if(j&0x1) sign = -1.0;

						complex<double> v = sign * Inner(d, j, k);

						A[r*m+c] = v;
					}
				}
			}
		}
	}
	return A;
}



// Journal of Computational Physics 227 (2008) 1836–1862
// Theorem 2
complex<double>* o2o_trans_mat(const int max_order, const monopole& d)
{
	int m = tensor_element_count(max_order);

	complex<double>* A = new complex<double>[m*m];

	for(int i=0; i<m*m; i++)
		A[i] = 0;

	for(int n=0; n<=max_order; n++)
	{
		for(int l=-n; l<=n; l++)
		{
			const int r = degree_order_to_index(n,l); //row
			for(int j=0; j<=max_order; j++)
			{
				for(int k=-j; k<=j; k++)
				{
					int src_degree = n+j;
					int src_order = l+k;
					const int c = degree_order_to_index(src_degree, src_order); //col
					if(c >= 0 && c < m) //then valid
					{
						double sign = 1.0;
						if(j&0x1) sign = -1.0;

						complex<double> v = sign * Inner(d, j, -k);

						A[r*m+c] = v;
					}
				}
			}
		}
	}
	return A;
}



