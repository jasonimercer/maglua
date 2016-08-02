#include "fmm_math.h"
#include <complex>
#include <math.h>
#include <stdio.h>
using namespace std;


static double faci(int x)
{
	if(x < 0 || x > 20)
		return 1;

//	double f = 1;
//	for(; x>1; x--)
//		f *= x;
//	return f;

	static const double faci_lookup[21] =
		{
		 1, // 0!
		 1,
		 2,
		 6,
		 24,
		 120, // 5!
		 720,
		 5040,
		 40320,
		 362880,
		 3628800, // 10!
		 39916800,
		 479001600,
		 6227020800,
		 87178291200,
		 1307674368000, //15!
		 20922789888000,
		 355687428096000,
		 6402373705728000,
		 121645100408832000,
		 2432902008176640000 //20!
		};
	return faci_lookup[x];
}



double negOnePow(const int l)
{
	if(l < 0)
		return 1.0;

	if( (l & 0x1) ) //odd
		return -1.0;
	return 1.0;
}

// hard coding gamma function for integers 1 to 20
long Gammai(long z)
{
	switch(z)
	{
	case  1: return  1;
	case  2: return  1;
	case  3: return  2;
	case  4: return  6;
	case  5: return  24;
	case  6: return  120;
	case  7: return  720;
	case  8: return  5040;
	case  9: return  40320;
	case 10: return  362880;
	case 11: return  3628800;
	case 12: return  39916800;
	case 13: return  479001600;
	case 14: return  6227020800;
	case 15: return  87178291200;
	case 16: return  1307674368000;
	case 17: return  20922789888000;
	case 18: return  355687428096000;
	case 19: return  6402373705728000;
	case 20: return  121645100408832000;
	}
	fprintf(stderr, "(%s:%i) No rule for Gamma(%i)\n", __FILE__, __LINE__, (int)z);
	return 0;
}

complex<double> Ylm(const int n, const int l, const double theta, const double phi)
{
	// Journal of Computational Physics 227 (2008) 1836–1862
	// "We emphasize that our spherical harmonics are considered as identically null for n < 0  or |l| > n."
	if(n < 0)
		return complex<double>(0,0);
	if(abs(l) > n)
		return complex<double>(0,0);
	//

	const double a = faci(n - l);
	const double b = faci(n + l);
	const double c = Plm(n, l, cos(theta));

	double S, C;
	sincos(((double)l) * phi, &S, &C);

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

void OutterTensor(const monopole& r, const int order, complex<double>* tensor)
{
	int c = 0;
	for(int n=0; n<=order; n++)
	{
		for(int l=-n; l<=n; l++)
		{
			tensor[c] = Outter(r, n, l);
			c++;
		}
	}
}

complex<double> im(const int v)
{
	return complex<double>(0,v);
}


// ii ^ x
// works for x > -400
complex<double> ipow(const int x)
{
	switch( (x+400) & 0x3 )
	{
	case 0:	return complex<double>( 1, 0);
	case 1:	return complex<double>( 0, 1);
	case 2:	return complex<double>(-1, 0);
	case 3:	return complex<double>( 0,-1);
	}
	return 0;
}

// There is an optimization opportunity here
// Pln0 and Pln1 have a lot of overlap, can cut down Pln function calls by about 1/2.
// turns out this opt doesn't save much. Look elsewhere for gains

// making these static. This breaks multithreading over gradOutterTensor
static complex<double> pln0[256]; //larger than needed
static complex<double> pln1[121];
static complex<double>   xy[121]; //common to x&y terms
static complex<double> eterm[40];
static complex<double>  __ir[12];
void gradOutterTensor(const monopole& R, const int max_degree, complex<double>* dx, complex<double>* dy, complex<double>* dz)
{
	if(max_degree > 9)
	{
		fprintf(stderr, "(%s:%i) max_degree > 9 not supported\n", __FILE__, __LINE__);
		return;
	}
	complex<double> ii(0,1);
	int count = tensor_element_count(max_degree);
	//int count1 = tensor_element_count(order+1);

#if 0
	complex<double>* pln0;
	complex<double>* pln1;
	complex<double>* xy;

	pln0 = new complex<double>[count];
	pln1 = new complex<double>[count];
	xy   = new complex<double>[count]; //common to x&y terms
#endif



	const double x = R.x;
	const double y = R.y;
	const double z = R.z;
	const double r = R.r;

	const double r2 = r*r;


	int c = 0;
	int cc;
	for(int n=0; n<=max_degree+1; n++) //order+1 so that we can get the n+1 terms calculated here
	{
#if 0
		for(int l=-n; l<=n; l++)
		{
			pln0[c] = Plm(n,   l, z/r);
			pln1[c] = Plm(n+1, l, z/r);
			//xy[c] = -exp(-0.5 * im(l) * (M_PI - 2 * R.p)) * pow(r2, -1.5 - 0.5*n) * faci(n-l) / (x*x + y*y);
			c++;
		}
#else
		// using calculated Plm to get the others
		for(int l=-n; l<=0; l++)
		{
			pln0[c] = Plm(n,   l, z/r);
			//pln1[c] = Plm(n+1, l, z/r);
			//xy[c] = -exp(-0.5 * im(l) * (M_PI - 2 * R.p)) * pow(r2, -1.5 - 0.5*n) * faci(n-l) / (x*x + y*y);
			c++;
		}
		cc = c-2;
		for(int l=1; l<=n; l++)
		{
			pln0[c] = Plm_negate_order(n,   l, pln0[cc].real());  //Plm(n,   l, z/r);
			//pln1[c] = Plm_negate_order(n+1, l, pln1[cc].real());//  Plm(n+1, l, z/r);
			c++;
			cc--;
		}
#endif
	}

	//read off n+1 terms
	cc = 0;
	c = 0;
	for(int n=0; n<=max_degree; n++) //order+1 so that we can get the n+1 terms calculated here
	{
		cc+=2;
		for(int l=-n; l<=n; l++)
		{
			pln1[c] = pln0[cc];
			c++;
			cc++;
		}
	}

	// Computing the xy[c] term piecewise and smarter
	eterm[20] = -1;
	for(int l=1; l<=max_degree; l++)
	{
		eterm[20 + l] = -exp(-0.5 * im(l) * (M_PI - 2 * R.p));
		eterm[20 - l] = 1.0 / eterm[20 + l];
	}

	__ir[0] = 1.0;
	__ir[1] = 1.0/r;
	const complex<double> ir2 = __ir[1]*__ir[1];
	for(int i=2; i<=max_degree+2; i+=2)
	{
		__ir[i+0] = __ir[i-2] * ir2;
		__ir[i+1] = __ir[i-1] * ir2;
	}

	const double ixxyy = 1.0 / (x*x+y*y);
	c = 0;
	for(int n=0; n<=max_degree; n++)
	{
		const double r2_term = pow(r2, -1.5 - 0.5*n);
		for(int l=-n; l<=n; l++)
		{
			xy[c] = eterm[20+l] * (r2_term * faci(n-l) * ixxyy);
			c++;
		}
	}

	c = 0;
	for(int n=0; n<=max_degree; n++)
	{
		for(int l=-n; l<=n; l++)
		{
			dx[c] = xy[c] * ((x+n*x+im(l)*y)*r2 * pln0[c] + (l-n-1)*x*z*r*pln1[c]);
			dy[c] = xy[c] * ((y+n*y-im(l)*x)*r2 * pln0[c] + (l-n-1)*y*z*r*pln1[c]);
			//dz[c] = -pow(ii,-l) * exp(im(l)*R.p) * pow(r2,-1-0.5*n) * (double)Gammai(2-l+n) * pln1[c];
			//dz[c] = -ipow(-l) * exp(im(l)*R.p) * pow(r2,-1-0.5*n) * (double)Gammai(2-l+n) * pln1[c];
			dz[c] = -ipow(-l) * exp(im(l)*R.p) * (__ir[n+2]) * (double)Gammai(2-l+n) * pln1[c];
			c++;
		}
	}

#if 0
	delete [] pln0;
	delete [] pln1;
	delete [] xy;
#endif
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

void InnerTensor(const monopole& r, const int order, complex<double>* tensor)
{
	int c = 0;
	for(int n=0; n<=order; n++)
	{
		for(int l=-n; l<=n; l++)
		{
			tensor[c] = r.q * pow(-1.0, n) * Inner(r, n, -l);
			c++;
		}
	}
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



#if 0
tensor_transformation::tensor_transformation(double x, double y, double z, int d)
{
    degree = d;
    t = i2i_trans_mat(degree, monopole(x,y,z));
}

tensor_transformation::~tensor_transformation()
{
    delete [] t;
}

void tensor_transformation::apply(const std::complex<double>* x, std::complex<double>* b) const
{
    const complex<double>* A = t;

    int m = tensor_element_count(degree);

    for(int r=0; r<m; r++)
        b[r] = 0;

    for(int r=0; r<m; r++)
        for(int c=0; c<m; c++)
        {
            b[r] += A[r*m+c] * x[c];
        }
}
#endif





int tensor_element_count(const int order)
{
	if(order < 0)
		return 0;

	return 1 + 2*order + order*order;

	//	return (2*order+1) + tensor_element_count(order-1);
}


static int degree_order_to_index(const int l, const int m)
{
	if(abs(m) > l)
		return -1;
	return tensor_element_count(l-1) + l + m;
}



void tensor_mat_mul_LowerTri(const complex<double>* A, const complex<double>* x, complex<double>* b, int max_degree)
{
    int m = tensor_element_count(max_degree);

	for(int r=0; r<m; r++)
	{
		b[r] = 0;
	}


	for(int r=0; r<m; r++)
	{
		//for(int c=0; c<m; c++)
		for(int c=0; c<=r; c++)
		{
			b[r] += A[r*m+c] * x[c];
		}
	}
}

complex<double> tensor_contract(const complex<double>* t1, const complex<double>* t2, const int len_not_max_degree)
{
	complex<double> sum = 0;
	for(int i=0; i<len_not_max_degree; i++)
	{
		sum += t1[i] * t2[i];
	}
	return sum;
}



// Journal of Computational Physics 227 (2008) 1836–1862
// Theorem 3
complex<double>* i2i_trans_mat(const int max_degree, const monopole& d, complex<double>* array)
{
    int m = tensor_element_count(max_degree);

	complex<double>* A;

	if(array)
		A = array;
	else
		A = new complex<double>[m*m];

	for(int i=0; i<m*m; i++)
		A[i] = 0;

	int r = 0;
    for(int n=0; n<=max_degree; n++)
	{
		for(int l=-n; l<=n; l++)
		{
			int c = 0;
            for(int j=0; j<=max_degree; j++)
			{
				for(int k=-j; k<=j; k++)
				{
					A[r*m+c] =  Inner(d, n-j, -l+k);
					c++;
				}
			}
			r++;
		}
	}
	return A;
}



// Journal of Computational Physics 227 (2008) 1836–1862
// Theorem 2
complex<double>* o2o_trans_mat(const int max_degree, const monopole& d)
{
    int m = tensor_element_count(max_degree);

	complex<double>* A = new complex<double>[m*m];

	for(int i=0; i<m*m; i++)
		A[i] = 0;

    for(int n=0; n<=max_degree; n++)
	{
		for(int l=-n; l<=n; l++)
		{
			const int r = degree_order_to_index(n,l); //row
            for(int j=0; j<=max_degree; j++)
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



