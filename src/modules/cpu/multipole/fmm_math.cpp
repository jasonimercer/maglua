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

int faci(int x)
{
	int f = 1;
	for(; x>1; x--)
		f *= x;
	return f;
}

int facfac(int x)
{
	if(! (x&0x1)) //then even
		x--; //need odd
	
	int f = 1;
	for(; x>1; x-=2)
		f *= x;
	return f;
}


double P00(const double x) {return 1;}

double P10(const double x) {return x;}
double P11(const double x) {return -1.0 * sqrt(1.0 - x*x);}

double P20(const double x) {return 0.5 * (3.0 * x*x - 1.0);}
double P21(const double x) {return -3.0 * x * sqrt(1.0 - x*x);}
double P22(const double x) {return 3.0 * (1.0 - x*x);}

double P30(const double x) {return 0.5 * x * (5.0 * x * x - 3.0);}
double P31(const double x) {return 1.5 * (1.0 - 5.0 * x * x) * sqrt(1.0-x*x);}
double P32(const double x) {return 15.0 * x * (1.0-x*x);}
double P33(const double x) {return -15.0 * pow(1.0 - x*x, 1.5);}
//   m     m <= l
// P      
//   l   
double Plm(const int l, const int m, const double x)
{
	// casting to double
	const double L = l;
	const double M = m;
	if(abs(m) > l)
		return 0;
	if(l < 0)
	{
		return Plm(l-1,m,x);
	}

	if(m < 0)
	{
		const int em = -m;
		double prefactor1 = 1;
		if(em & 2)
			prefactor1 = -1;
		double prefactor2a = faci(l - em);
		double prefactor2b = faci(l + em);
		return prefactor1 * (prefactor2a / prefactor2b) * Plm(l,em,x);
	}
	if(l <= 3 && m <= l)
	{
		if(l == 3)
		{
			if(m == 0) return P30(x);
			if(m == 1) return P31(x);
			if(m == 2) return P32(x);			
			if(m == 3) return P33(x);			
		}
		if(l == 2)
		{
			if(m == 0) return P20(x);
			if(m == 1) return P21(x);
			if(m == 2) return P22(x);			
		}
		if(l == 1)
		{
			if(m == 0) return P10(x);
			if(m == 1) return P11(x);
		}
		if(l == 0)
		{
			if(m == 0) return P00(x);
		}
		fprintf(stderr, "(%s:%i) l = %i   m = %i    something went wrong\n", __FILE__, __LINE__, l,m);
		return 0;
	}
	
	if(l == m)
	{
		const double v = facfac(2*l-1) * pow(1.0-x*x, L/2.0);
		if(l & 0x1) //odd
			return -1.0 * v;
		return v;
	}
	
	if( (l-1) == m)
	{
		return x * (2.0*M+1.0) * Plm(m,m,x);
	}
	
	// Must: m <= l
	if(l - m >= 2) //then reduce l
	{
		return (x * (2.0*L-1.0)*Plm(l-1,m,x) - (L+M-1.0)*Plm(l-2,m,x) ) /
				(L - M);
	}
	else // reduce m
	{
		// todo: see if there is a problem when x = 1
		return (2.0*(M-1.0)*x) / (sqrt(1.0-x*x)) * Plm(l,m-1,x) +
			   ((M-1)*(M-2) - L * (L+1)) * Plm(l,m-2,x);
	}
}

complex<double> Ylm(const int l, const int m, const double phi, const double theta)
{
	double a = 1;
	if(m & 0x1)
		a = -1.0;
	
	const double b = (2.0 * ((double)l) + 1.0) / (4.0 * M_PI);
	const double c = ((double)faci(l-m)) / ((double)faci(l+m));
	const double d = Plm(l,m,cos(theta));

	void sincos(double x, double *sin, double *cos);

	//exp(i x) = cos(x) + i sin(x)
	double S, C;
	sincos(((double)m) * phi, &S, &C);
	
	const double f = a * sqrt(b*c) * d;
	
	return complex<double>(f*C, f*S);
}


double Tlm(const int l, const int m, const double theta)
{
	return sqrt(	((double)(2*l+1) * faci(l-m))) / 
				((double)(4.0*M_PI* faci(l+m)))  * Plm(l,m,theta);
}

double Phim(const int m, double phi)
{
	if(m >  0)
		return SQRT2 * cos( ((double)m) * phi );
	if(m == 0)
		return 1;
	return SQRT2 * sin( fabs(m) * phi );
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



// equation 2 of IEEE Trans Mag 37. 3
complex<double> Flm_Plm_r2(int l, int m, double r2, double _Plm, double phi)
{
	const double L = l;
	const double M = m;
	const double f1 = faci(l-m);
	const double f2 = _Plm;
	const double d1 = pow(r2, (L+1.0)/2.0);

	const double value = f1*f2/d1;

	//exp(i x) = cos(x) + i sin(x)
	double S, C;
	sincos(M * phi, &S, &C);
	return complex<double>(value*C, value*S);
}

complex<double> Flm_Plm(int l, int m, double rx, double ry, double rz, double _Plm, double phi)
{
	return Flm_Plm_r2(l,m,rx*rx+ry*ry+rz*rz,_Plm,phi);
}
complex<double> Flm(int l, int m, double rx, double ry, double rz, double cos_theta, double phi)
{
	return Flm_Plm(l,m,rx,ry,rz, Plm(l,m,cos_theta), phi);
}


// equation 3 of IEEE Trans Mag 37. 3
complex<double> Nlm_Plm_r2(int l, int m, double r2, double _Plm, double phi)
{
	const double L = l;
	const double M = m;
	const double f1 = pow(r2, L/2.0);
	const double f2 = _Plm;
	const double d1 = faci(l+m);

	const double value = f1*f2/d1;

	double S, C;
	sincos(-1.0 * M * phi, &S, &C);
	return complex<double>(value*C, value*S);
}

complex<double> Nlm_Plm(int l, int m, double rx, double ry, double rz, double _Plm, double phi)
{
	return Nlm_Plm_r2(l,m,rx*rx+ry*ry+rz*rz,_Plm,phi);
}
complex<double> Nlm(int l, int m, double rx, double ry, double rz, double cos_theta, double phi)
{
	return Nlm_Plm(l,m,rx,ry,rz, Plm(l,m,cos_theta), phi);
}





// equation 6 of IEEE Trans Mag 37. 3
complex<double> Contract_Flm_Nlm(int lmax, double rx, double ry, double rz, double cos_theta, double phi)
{
	const double r2 = rx*rx+ry*ry+rz*rz;
	complex<double> sum = 0;
	for(int l=0; l<=lmax; l++)
	{
		for(int m=-l; m<=l; l++)
		{
			const double _Plm = Plm(l,m,cos_theta);
			const complex<double> f = Flm_Plm_r2(l,m,r2,_Plm,phi);
			const complex<double> n = Nlm_Plm_r2(l,m,r2,_Plm,phi);
			sum += f*n;
		}
	}
	return sum;
}




void negGradNlm(int l, int m, const double r, const double cos_theta, const double sin_theta, const double phi, complex<double>* res3)
{
	const double L = l;
	const double M = m;

	const double cot_theta = cos_theta / sin_theta;
	const double csc_theta =       1.0 / sin_theta;

	const complex<double> N = Nlm_Plm_r2(l, m, r*r, Plm(l,m,cos_theta), phi);
	const complex<double> Nl1 = Nlm_Plm_r2(l+1, m, r*r, Plm(l,m,cos_theta), phi);
	const complex<double> phi_scale = complex<double>(0, M / (r * sin_theta));

	res3[0] = -(L / r) * N;
	res3[1] = phi_scale * N;
	res3[2] = (-1.0/r) * (csc_theta * (M-L-1.0) * Nl1 + cot_theta * (L+1.0) * N);
}

void negGradFlm(int l, int m, const double r, const double cos_theta, const double sin_theta, const double phi, complex<double>* res3)
{
	const double L = l;
	const double M = m;

	const double cot_theta = cos_theta / sin_theta;
	const double csc_theta =       1.0 / sin_theta;

	const complex<double> F = Flm_Plm_r2(l, m, r*r, Plm(l,m,cos_theta), phi);
	const complex<double> Fl1 = Flm_Plm_r2(l+1, m, r*r, Plm(l,m,cos_theta), phi);
	const complex<double> phi_scale = complex<double>(0,-M / (r * sin_theta));

	res3[0] = ((L+1) / r) * F;
	res3[1] = phi_scale * F;
	res3[2] = (1.0/r) * (cot_theta * (L+1.0) * F + (M-L-1.0) * csc_theta * Fl1);
}




double PhiSingle1(double q, double* r3, double* r03)
{
	double r2 = pow(r3[0] - r03[0], 2);
	r2 += pow(r3[1] - r03[1], 2);
	r2 += pow(r3[2] - r03[2], 2);

	return q / (sqrt(r2) * 4.0 * M_PI);
}

std::complex<double> PhiSingle1(double q, double* r3, double* r03, int order)
{
	double t, t0, p, p0, r, r0;

	physics_spherical_coords(r3,  t,  p,  r);
	physics_spherical_coords(r03, t0, p0, r0);

	complex<double> sum = 0;

	for(int l=0; l<=order; l++)
	{
		for(int m=-l; m<=l; m++)
		{
			double _Plm = Plm(l, m, cos(t));
			double _Plm0 = Plm(l, m, cos(t0));

			const complex<double> a = Nlm_Plm_r2(l, m, r0*r0, _Plm0, p0) * Flm_Plm_r2(l, m, r*r, _Plm, p);
			printf("% 4i % 4i % 10e % 10e\n", l, m, a.real(), a.imag());
			sum +=a;
		}
	}

	sum *= (q / (4.0 * M_PI));
	return sum;
}

int tensor_element_count(const int order)
{
	if(order <= 0)
		return 1;
	return (2*order+1) + tensor_element_count(order-1);
}

void CF_tensor(const vector<monopole>& r, const int order, std::complex<double>* tensor)
{
	int c = 0;
	for(int l=0; l<=order; l++)
	{
		for(int m=-l; m<=l; m++)
		{
			complex<double> a = 0;
			for(unsigned int i=0; i<r.size(); i++)
			{
				complex<double> v = r[i].q * Nlm(l, m, r[i].x, r[i].y, r[i].z, cos(r[i].t), r[i].p);
				a += v;
			}
			tensor[c] = a / (4.0 * M_PI);
			c++;
		}
	}
}

void F_tensor(const monopole& r, const int order, std::complex<double>* tensor)
{
	int c = 0;
	for(int l=0; l<=order; l++)
	{
		for(int m=-l; m<=l; m++)
		{
			tensor[c] = Flm(l, m, r.x, r.y, r.z, cos(r.t), r.p);
			c++;
		}
	}
}

// rank 4 tensor. tensor_element_count(order)^2 elements
void TCF_tensor(const monopole& d, const int order, std::complex<double>* tensor)
{
	monopole nd(-d.x, -d.y, -d.z);
	int c = 0;
	for(int l=0; l<=order; l++)
	{
		for(int m=-l; m<=l; m++)
		{
			for(int i=0; i<=order; i++)
			{
				for(int j=-i; j<=i; j++)
				{
					tensor[c] = Nlm(l-i, m-j, nd.x, nd.y, nd.z, cos(nd.t), nd.p);
					c++;
				}
			}
		}
	}
}
// rank 4 tensor. tensor_element_count(order)^2 elements
void TCN_tensor(const monopole& d, const int order, std::complex<double>* tensor)
{
	int c = 0;
	for(int l=0; l<=order; l++)
	{
		for(int m=-l; m<=l; m++)
		{
			for(int i=0; i<=order; i++)
			{
				for(int j=-i; j<=i; j++)
				{
					tensor[c] = Nlm(i-l, j-m, d.x, d.y, d.z, cos(d.t), d.p);
					c++;
				}
			}
		}
	}
}
// rank 4 tensor. tensor_element_count(order)^2 elements
void TCNCF_tensor(const monopole& d, const int order, std::complex<double>* tensor)
{
	int c = 0;
	for(int l=0; l<=order; l++)
	{
		for(int m=-l; m<=l; m++)
		{
			for(int i=0; i<=order; i++)
			{
				for(int j=-i; j<=i; j++)
				{
					tensor[c] = pow(-1.0, l) * Flm(i+l, j+m, d.x, d.y, d.z, cos(d.t), d.p);
					c++;
				}
			}
		}
	}
}


void contract22_tensor(const int order, std::complex<double>& dest, std::complex<double>* src1, std::complex<double>* src2)
{
	int n = tensor_element_count(order);

	dest = 0;
	for(int i=0; i<n; i++)
	{
		dest += src1[i] * src2[i];
	}
}

void contract42_tensor(const int order, std::complex<double>* dest, std::complex<double>* src4, std::complex<double>* src2)
{
	int n = tensor_element_count(order);

	for(int j=0; j<n; j++)
	{
		dest[j] = 0;
	}

	for(int j=0; j<n; j++)
	{
		for(int i=0; i<n; i++)
		{
			dest[j] += src4[i+j*n] * src2[i];
		}
	}
}



