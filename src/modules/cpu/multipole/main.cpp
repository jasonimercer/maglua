#include "fmm_math.h"
#include <stdio.h>
#include <stdlib.h>

using namespace std;

//int tensor_element_count(const int order);

complex<double>* new_tensor(int order)
{
	return new complex<double>[ tensor_element_count(order) ];
}


int main(int argc, char** argv)
{
	int order = 5;

	monopole r(0.1,0,0);
	monopole x1(3,0,0);
	monopole x2(3,1,0);
	monopole x3(3,-2,0);

	monopole pos(0,20,1);

	complex<double> sum = 0;
	int c;
	int len = tensor_element_count(order);

	printf("Outter Goal: %g\n", 1.0/(pos-x1).r + 1.0/(pos-x2).r + 1.0/(pos-x3).r );

	complex<double>* t1 = new_tensor(order);
	complex<double>* t2 = new_tensor(order);
	complex<double>* t3 = new_tensor(order);
	complex<double>* tSum = new_tensor(order);
	complex<double>* tSum2= new_tensor(order);
	complex<double>* tp = new_tensor(order);

	// ===============================================================================
	// "standard"
	c = 0;
	for(int n=0; n<=order; n++)
	{
        double a = pow(-1.0, n);
		for(int l=-n; l<=n; l++)
		{
            t1[c] = Inner(x1, n, -l);
            t2[c] = Inner(x2, n, -l);
            t3[c] = Inner(x3, n, -l);
			tp[c] = Outter(pos, n, l);

            tSum[c] = a*(t1[c] + t2[c] + t3[c]);
            c++;
		}
	}

	sum = 0;
	for(int i=0; i<len; i++)
	{
		complex<double> a = tSum[i] * tp[i];
//        printf("%g %g\n", t1[i].real(), t1[i].imag());
		sum += a;
	}
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






//	free_fmm_rules();

	return 0;
}
