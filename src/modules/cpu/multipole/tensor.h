#ifndef TENSOR_H
#define TENSOR_H

#include "fmm_math.h"
using namespace std;

class Tensor
{
public:
	Tensor(int _dims=2) : dims(_dims) {}
};

class Flm : public Tensor
{
public:
	Flm() : Tensor(2) {}

	static complex<double> eval(int l, int m, double rx, double ry, double rz, double theta, double phi);
};

#endif // TENSOR_H
