// Jason Mercer 2012
// Function to help with Fast Multipole Method

#include <complex>
#include <vector>

class monopole
{
public:
	monopole(double _x=0, double _y=0, double _z=0, double _q=0) : x(_x), y(_y), z(_z), q(_q) {calcSpherical();}
	monopole(const monopole& p) : x(p.x), y(p.y), z(p.z), q(p.q) {calcSpherical();}

	void calcSpherical();
	double x, y, z, q;
	double t, p, r;

	monopole&  operator+=(const monopole& rhs);
	monopole&  operator-=(const monopole& rhs);

	const monopole operator+(const monopole& b) const {monopole c(*this); c+=b; return c;}
	const monopole operator-(const monopole& b) const	{monopole c(*this); c-=b; return c;}
};


double Plm(const int l, const int m, const double x);
std::complex<double> Ylm(const int l, const int m, const double phi, const double theta);


std::complex<double> Flm(int l, int m, double rx, double ry, double rz, double cos_theta, double phi);
std::complex<double> Flm_Plm(int l, int m, double rx, double ry, double rz, double _Plm, double phi);
std::complex<double> Flm_Plm_r2(int l, int m, double r2, double _Plm, double phi);

std::complex<double> Nlm(int l, int m, double rx, double ry, double rz, double cos_theta, double phi);
std::complex<double> Nlm_Plm(int l, int m, double rx, double ry, double rz, double _Plm, double phi);
std::complex<double> Nlm_Plm_r2(int l, int m, double r2, double _Plm, double phi);


std::complex<double> Contract_Flm_Nlm(int lmax, double rx, double ry, double rz, double cos_theta, double phi);

void negGradNlm(int l, int m, const double r, const double theta, const double phi, std::complex<double>* res3);


//Testing Functions
double PhiSingle1(double q, double* r3, double* r03);
std::complex<double> PhiSingle1(double q, double* r3, double* r03, int order);


int tensor_element_count(const int order);

// rank 2 tensors. tensor_element_count(order) elements
void CF_tensor(const std::vector<monopole>& r, const int order, std::complex<double>* tensor);
void F_tensor(const monopole& r, const int order, std::complex<double>* tensor);
void TCF_tensor(const monopole& d, const int order, std::complex<double>* tensor);

// rank 4 tensors. tensor_element_count(order)^2 elements
void TCF_tensor(const monopole& d, const int order, std::complex<double>* tensor);
void TCN_tensor(const monopole& d, const int order, std::complex<double>* tensor);
void TCNCF_tensor(const monopole& d, const int order, std::complex<double>* tensor);


void contract22_tensor(const int order, std::complex<double>& dest, std::complex<double>* src1, std::complex<double>* src2);
void contract42_tensor(const int order, std::complex<double>* dest, std::complex<double>* src4, std::complex<double>* src2);

