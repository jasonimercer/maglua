#include "fmm_math.h"
#include <stdio.h>
#include <stdlib.h>

using namespace std;

//int tensor_element_count(const int order);

complex<double>* new_tensor(int order)
{
	return new complex<double>[ tensor_element_count(order) ];
}

void dip(double rx, double ry, double rz, double mx, double my, double mz, double& hx, double& hy, double& hz)
{
	double mdotr = rx*mx+ry*my+rz*mz;
	double r = sqrt(rx*rx+ry*ry+rz*rz);
	double r3 = r*r*r;
	double r5 = r*r*r3;

	hx = 3.0 * rx * mdotr / r5 - mx / r3;
	hy = 3.0 * ry * mdotr / r5 - my / r3;
	hz = 3.0 * rz * mdotr / r5 - mz / r3;
}

int main(int argc, char** argv)
{
	int order = 8;

	monopole m(1,1,1);
	m.makeUnit();

	const double e = 1e-8;
	monopole r(0.1,0,0);
	monopole x1(3,0, 0,-0.5);
	monopole x2(3,0, 0, 0.5);
	x1 += m*e;
	x2 -= m*e;

	monopole pos(10,0,0);


	monopole dd[6];
	dd[0] = monopole( e, 0, 0);
	dd[1] = monopole(-e, 0, 0);
	dd[2] = monopole( 0, e, 0);
	dd[3] = monopole( 0,-e, 0);
	dd[4] = monopole( 0, 0, e);
	dd[5] = monopole( 0, 0,-e);


	double hx, hy, hz;

	monopole R = (x1 + x2)*0.5 - pos;
	printf("x1 %g %g %g\n", x1.x, x1.y, x1.z);
	printf("x2 %g %g %g\n", x2.x, x2.y, x2.z);
	printf("pos %g %g %g\n", pos.x, pos.y, pos.z);
	printf("R %g %g %g\n", R.x, R.y, R.z);
	dip(R.x, R.y, R.z, m.x, m.y, m.z, hx,hy,hz);

	printf("Hdip = %g %g %g\n", hx,hy,hz);


	complex<double> sum[6];
	int c;
	int len = tensor_element_count(order);

	//printf("Outter Goal: %g\n", x1.q/(pos-x1).r + x2.q/(pos-x2).r );

	complex<double>* t1 = new_tensor(order);
	complex<double>* t2 = new_tensor(order);
	complex<double>* tSum = new_tensor(order);

	complex<double>* tp[6];
	for(int i=0; i<6; i++) tp[i] = new_tensor(order);


	c = 0;
	for(int n=0; n<=order; n++)
	{
		for(int l=-n; l<=n; l++)
		{
			for(int i=0; i<6; i++)
				tp[i][c] = Outter(pos+dd[i], n, l);
			c++;
		}
	}


	// ===============================================================================
	// "standard"
	c = 0;
	for(int n=0; n<=order; n++)
	{
        double a = pow(-1.0, n);
		for(int l=-n; l<=n; l++)
		{
			t1[c] = x1.q * Inner(x1, n, -l);
			t2[c] = x2.q * Inner(x2, n, -l);

			tSum[c] = a*(t1[c] + t2[c]);
            c++;
		}
	}

	for(int i=0; i<6; i++)
	{
		sum[i] = 0;
		for(int j=0; j<len; j++)
		{
			complex<double> a = tSum[j] * tp[i][j];
			sum[i] += a;
		}
	}

	hx = (sum[0] - sum[1]).real() / (2.0*e*e);
	hy = (sum[2] - sum[3]).real() / (2.0*e*e);
	hz = (sum[4] - sum[5]).real() / (2.0*e*e);

	printf("       %g %g %g\n", hx, hy, hz);

#if 0
//    return 0;

	printf("standard:    %g %g\n", sum.real(), sum.imag());
	// ===============================================================================




	// ===============================================================================
	// "shifted"
	c = 0;
	for(int n=0; n<order; n++)
	{
		double a = pow(-1.0, n);
		for(int l=-n; l<=n; l++)
		{
            t1[c] = Inner(x1+r, n, -l);
            t2[c] = Inner(x2+r, n, -l);
            t3[c] = Inner(x3+r, n, -l);
			tp[c] = Outter(pos+r, n, l);

            tSum[c] = a*(t1[c] + t2[c] + t3[c]);
			c++;
		}
	}

	sum = 0;
	for(int i=0; i<len; i++)
	{
		sum += tSum[i] * tp[i];
	}

	printf("shifted:     %g %g\n", sum.real(), sum.imag());
	// ===============================================================================




	// ===============================================================================
	// "standard" + translation
	c = 0;
	for(int n=0; n<order; n++)
	{
		double a = pow(-1.0, n);
		for(int l=-n; l<=n; l++)
		{
			t1[c] = a*Inner(x1, n, -l);
			t2[c] = a*Inner(x2, n, -l);
			t3[c] = a*Inner(x3, n, -l);
			tp[c] = Outter(pos-r, n, l);

			tSum[c] = t1[c] + t2[c] + t3[c];
			c++;
		}
	}

	complex<double>* trans = i2i_trans_mat(order, -r);
	tensor_mat_mul(trans, tSum, tSum2, order);


	sum = 0;
	for(int i=0; i<len; i++)
	{
		sum += tSum2[i] * tp[i];
	}

	printf("s+trans:     %g %g\n", sum.real(), sum.imag());
	// ===============================================================================










	monopole origin(0.1,0.1,0.1);
	printf("\n\nInner Goal: %g\n", 1.0/(origin-x1).r + 1.0/(origin-x2).r + 1.0/(origin-x3).r );

	// ===============================================================================
	// "standard"
	c = 0;
	for(int n=0; n<=order; n++)
	{
		double a = pow(-1.0, n);
		for(int l=-n; l<=n; l++)
		{
			t1[c] = a*Outter(x1, n, l);
			t2[c] = a*Outter(x2, n, l);
			t3[c] = a*Outter(x3, n, l);
			tp[c] = Inner(origin, n, -l);

			tSum[c] = t1[c] + t2[c] + t3[c];
			c++;
		}
	}

	sum = 0;
	for(int i=0; i<len; i++)
	{
		complex<double> a = tSum[i] * tp[i];
		sum += a;
	}

	complex<double> a = Outter(origin, 0, 0);


	printf("standard:   %g %g\n", sum.real(), sum.imag());
	// ===============================================================================





	// ===============================================================================
	// "shifted"
	c = 0;
	for(int n=0; n<order; n++)
	{
		double a = pow(-1.0, n);
		for(int l=-n; l<=n; l++)
		{
			t1[c] = a*Outter(x1+r, n, l);
			t2[c] = a*Outter(x2+r, n, l);
			t3[c] = a*Outter(x3+r, n, l);
			tp[c] = Inner(origin+r, n, -l);

			tSum[c] = t1[c] + t2[c] + t3[c];
			c++;
		}
	}

	sum = 0;
	for(int i=0; i<len; i++)
	{
		sum += tSum[i] * tp[i];
	}

	printf("shifted:    %g %g\n", sum.real(), sum.imag());
	// ===============================================================================





	// ===============================================================================
	// "standard" + translation
	c = 0;
	for(int n=0; n<order; n++)
	{
		double a = pow(-1.0, n);
		for(int l=-n; l<=n; l++)
		{
			t1[c] = a*Outter(x1, n, l);
			t2[c] = a*Outter(x2, n, l);
			t3[c] = a*Outter(x3, n, l);
			tp[c] = Inner(origin+r, n, -l);

			tSum[c] = t1[c] + t2[c] + t3[c];
			c++;
		}
	}

	complex<double>* trans2 = o2o_trans_mat(order, r);
	tensor_mat_mul(trans2, tSum, tSum2, order);

	sum = 0;
	for(int i=0; i<len; i++)
	{
		sum += tSum2[i] * tp[i];
	}

	printf("s+trans:    %g %g\n", sum.real(), sum.imag());
	// ===============================================================================



#endif
	return 0;
}
