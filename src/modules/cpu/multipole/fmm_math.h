// Jason Mercer 2012
// Function to help with Fast Multipole Method

#include <complex>
#include <vector>
#include "plm.h"

#ifndef MULTIPOLE_CLASSES
#define MULTIPOLE_CLASSES

class monopole
{
public:
	monopole(double _x=0, double _y=0, double _z=0, double _q=0) : x(_x), y(_y), z(_z), q(_q) {calcSpherical();}
	monopole(double* v3, double _q=0) {x=v3[0]; y=v3[1], z=v3[2], q=_q, calcSpherical();}
	monopole(const monopole& p) : x(p.x), y(p.y), z(p.z), q(p.q) {calcSpherical();}

	void calcSpherical();
	double x, y, z, q;
	double t, p, r;

	monopole&  operator+=(const monopole& rhs);
	monopole&  operator-=(const monopole& rhs);
	monopole&  operator*=(const double rhs);

	const monopole operator+(const monopole& b) const {monopole c(*this); c+=b; return c;}
	const monopole operator-(const monopole& b) const	{monopole c(*this); c-=b; return c;}
	const monopole operator*(const double b) const	{monopole c(*this); c*=b; return c;}
	const monopole operator/(const double b) const	{monopole c(*this); c*=(1.0/b); return c;}
	const monopole operator-() const {monopole c(*this); c*=-1.0; return c;}

	void makeUnit() {if(r != 0) {x/=r; y/=r; z/=r; r=1;} else {x=1; calcSpherical();}}
};

#if 0
class tensor_transformation
{
public:
    tensor_transformation(double x, double y, double z, int degree);
    ~tensor_transformation();

    void apply(const std::complex<double>* x, std::complex<double>* b) const;

    double rx, ry, rz;
    int degree;
    std::complex<double>* t;
};
#endif

#endif


#ifndef FMM_FUNCS
#define FMM_FUNCS
std::complex<double> Ylm(const int l, const int m, const double theta, const double phi);

int tensor_element_count(const int degree);

// Journal of Computational Physics 227 (2008) 1836–1862
std::complex<double> Inner(const monopole& r, int n, int l);
std::complex<double> Outter(const monopole& r, int n, int l);

void InnerTensor(const monopole& r, const int order, std::complex<double>* tensor);
void OutterTensor(const monopole& r, const int order, std::complex<double>* tensor);

void gradOutterTensor(const monopole& R, const int max_degree, std::complex<double>* dx, std::complex<double>* dy, std::complex<double>* dz);


// if no dest if given then memory will be allocated and returned otherwise the given memory will be used and returned
std::complex<double>* i2i_trans_mat(const int max_order, const monopole& d, std::complex<double>* dest=0);
std::complex<double>* o2o_trans_mat(const int max_order, const monopole& d);

void tensor_mat_mul_LowerTri(const std::complex<double>* A, const std::complex<double>* x, std::complex<double>* b, int max_degree);
std::complex<double> tensor_contract(const std::complex<double>* t1, const std::complex<double>* t2, const int len_not_max_degree);
#endif
