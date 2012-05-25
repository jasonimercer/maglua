#include "fmm_math.h"
#include <stdio.h>

using namespace std;

int main(int argc, char** argv)
{
	int order = 6;
	vector<monopole> charges;

	charges.push_back(monopole(1,1,1, 1));
	charges.push_back(monopole(1,0,0, 1));
	charges.push_back(monopole(0,1,0,-1));

	monopole d(1,2,3);
//	monopole d(0,0,0);
	monopole pos(10,5,2);
	pos += d;

	int len = tensor_element_count(order);

	complex<double>* CF = new complex<double>[len];
	complex<double>* CFnew = new complex<double>[len];
	complex<double>* F  = new complex<double>[len];
	complex<double>* TCF = new complex<double>[len*len];


	CF_tensor(charges, order, CF);
	F_tensor(pos, order, F);
	TCF_tensor(d, order, TCF);

	contract42_tensor(order, CFnew, TCF, CF);


	complex<double> result;
	contract22_tensor(order, result, CFnew, F);

	printf("%g %g\n", result.real(), result.imag());


	pos -= d;
	double r0[3];
	r0[0] = pos.x;
	r0[1] = pos.y;
	r0[2] = pos.z;
	double p = 0;
	for(unsigned int i=0; i<3; i++)
	{
		double r[3];
		r[0] = charges[i].x;
		r[1] = charges[i].y;
		r[2] = charges[i].z;

		p += PhiSingle1(charges[i].q, r, r0);
	}
	printf("%g\n", p);

	printf("diff: %g\n", fabs(p - result.real()));

	return 0;
}
