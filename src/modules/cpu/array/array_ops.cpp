#include <math.h>
#include "array_ops.h"
#include <stdio.h>


template<typename T>
void setAll_(T* dest, const int n, const T& v)
{
	for(int i=0; i<n; i++)
		dest[i] = v;
}

void arraySetAll(double* a, const int n, const double& v)
{
	setAll_<double>(a, n, v);
}
void arraySetAll(float* a, const int n, const float& v)
{
	setAll_<float>(a, n, v);
}
void arraySetAll(int* a, const int n, const int& v)
{
	setAll_<int>(a, n, v);
}
void arraySetAll(doubleComplex* a, const int n, const doubleComplex& v)
{
	setAll_<doubleComplex>(a, n, v);
}
void arraySetAll(floatComplex* a, const int n, const floatComplex& v)
{
	setAll_<floatComplex>(a, n, v);
}




template<typename T>
void scaleAll_(T* dest, const int n, const T& v)
{
	for(int i=0; i<n; i++)
		dest[i] *= v;
}
void arrayScaleAll(double* a, const int n, const double& v)
{
	scaleAll_<double>(a, n, v);
}
void arrayScaleAll(float* a, const int n, const float& v)
{
	scaleAll_<float>(a, n, v);
}
void arrayScaleAll(int* a, const int n, const int& v)
{
	scaleAll_<int>(a, n, v);
}
void arrayScaleAll(doubleComplex* a, const int n, const doubleComplex& v)
{
	scaleAll_<doubleComplex>(a, n, v);
}
void arrayScaleAll(floatComplex* a, const int n, const floatComplex& v)
{
	scaleAll_<floatComplex>(a, n, v);
}





// dest[i] = (mult1 * src1[i] + add1) * (mult2 * src2[i] + add2)
template<typename T>
void arrayDot_(T* dest, T* s1, T* s2, const int n)
{
	for(int i=0; i<n; i++)
		dest[i] = s1[i] * s2[i];
}


void arrayMultAll(double* d, double* s1, double* s2, const int n)
{
	arrayDot_<double>(d, s1, s2, n);
}
void arrayMultAll(float* d, float* s1, float* s2, const int n)
{
	arrayDot_<float>(d, s1, s2, n);
}
void arrayMultAll(int* d, int* s1, int* s2, const int n)
{
	arrayDot_<int>(d, s1, s2, n);
}
void arrayMultAll(doubleComplex* d, doubleComplex* s1, doubleComplex* s2, const int n)
{
	arrayDot_<doubleComplex>(d, s1, s2, n);
}
void arrayMultAll(floatComplex* d, floatComplex* s1, floatComplex* s2, const int n)
{
	arrayDot_<floatComplex>(d, s1, s2, n);
}








template<typename T>
void arrayDiff_(T* dest, T* s1, T* s2, const int n)
{
	for(int i=0; i<n; i++)
		dest[i] = s1[i] - s2[i];
}

void arrayDiffAll(double* d, double* s1, double* s2, const int n)
{
	arrayDiff_<double>(d, s1, s2, n);
}
void arrayDiffAll(float* d, float* s1, float* s2, const int n)
{
	arrayDiff_<float>(d, s1, s2, n);
}
void arrayDiffAll(int* d, int* s1, int* s2, const int n)
{
	arrayDiff_<int>(d, s1, s2, n);
}
void arrayDiffAll(doubleComplex* d, doubleComplex* s1, doubleComplex* s2, const int n)
{
	arrayDiff_<doubleComplex>(d, s1, s2, n);
}
void arrayDiffAll(floatComplex* d, floatComplex* s1, floatComplex* s2, const int n)
{
	arrayDiff_<floatComplex>(d, s1, s2, n);
}







void arrayNormAll(double* d, double* s1, const int n)
{
	for(int i=0; i<n; i++)
		d[i] = fabs(s1[i]);
}
void arrayNormAll(float* d, float* s1, const int n)
{
	for(int i=0; i<n; i++)
		d[i] = fabsf(s1[i]);
}
void arrayNormAll(int* d, int* s1, const int n)
{
	for(int i=0; i<n; i++)
		d[i] = abs(s1[i]);
}
void arrayNormAll(doubleComplex* d, doubleComplex* s1, const int n)
{
	for(int i=0; i<n; i++)
		d[i] = doubleComplex(abs(s1[i]), 0);
}
void arrayNormAll(floatComplex* d, floatComplex* s1, const int n)
{
	for(int i=0; i<n; i++)
		d[i] = floatComplex(abs(s1[i]), 0);
}





template<typename A, typename B>
void arrayGetRealPart_(A* dest, const B* src, const int n)
{
	for(int i=0; i<n; i++)
		dest[i] = src[i].real();
}
template<typename A, typename B>
void arrayGetImagPart_(A* dest, const B* src, const int n)
{
	for(int i=0; i<n; i++)
		dest[i] = src[i].imag();
}




void arrayGetRealPart(double* dest, const doubleComplex* src, const int n)
{
	arrayGetRealPart_<double,doubleComplex>(dest, src, n);
}
void arrayGetRealPart(float* dest, const floatComplex* src, const int n)
{
	arrayGetRealPart_<float,floatComplex>(dest, src, n);
}

void arrayGetImagPart(double* dest, const doubleComplex* src, const int n)
{
	arrayGetImagPart_<double,doubleComplex>(dest, src, n);
}
void arrayGetImagPart(float* dest, const floatComplex* src, const int n)
{
	arrayGetImagPart_<float,floatComplex>(dest, src, n);
}


template<typename A, typename B>
void arraySetRealPart_(A* dest, const B* src, const int n)
{
	for(int i=0; i<n; i++)
		dest[i] = A(src[i], dest[i].imag());
}
template<typename A, typename B>
void arraySetImagPart_(A* dest, const B* src, const int n)
{
	for(int i=0; i<n; i++)
		dest[i] = A(dest[i].real(), src[i]);
}



void arraySetRealPart(doubleComplex* dest, const double * src, const int n)
{
	arraySetRealPart_<doubleComplex,double>(dest, src, n);
}
void arraySetRealPart(floatComplex* dest, const float * src, const int n)
{
	arraySetRealPart_<floatComplex,float>(dest, src, n);
}

void arraySetImagPart(doubleComplex* dest, const double * src, const int n)
{
	arraySetImagPart_<doubleComplex,double>(dest, src, n);
}
void arraySetImagPart(floatComplex* dest, const float * src, const int n)
{
	arraySetImagPart_<floatComplex,float>(dest, src, n);
}




template<typename T>
T arraySumAll_(T* v, const int n)
{
	T res = 0;
	for(int i=0; i<n; i++)
		res += v[i];
	
	return res;
}



void arraySumAll(double* a, const int n, double& v)
{
	v = arraySumAll_<double>(a, n);
}
void arraySumAll(float* a, const int n, float& v)
{
	v = arraySumAll_<float>(a, n);
}
void arraySumAll(int* a, const int n, int& v)
{
	v = arraySumAll_<int>(a, n);
}
void arraySumAll(doubleComplex* a, const int n, doubleComplex& v)
{
	v = arraySumAll_<doubleComplex>(a, n);
}
void arraySumAll(floatComplex* a, const int n, floatComplex& v)
{
	v = arraySumAll_<floatComplex>(a, n);
}







