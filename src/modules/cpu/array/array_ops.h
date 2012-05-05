#include <complex>
#ifndef COMPLEX_TYPES
#define COMPLEX_TYPES
using namespace std;
typedef complex<double> doubleComplex; //cpu version
typedef complex<float>   floatComplex;
#endif

#ifndef ARRAY_SUPPORT_FUNCTIONS
#define ARRAY_SUPPORT_FUNCTIONS


void arraySetAll(double* d_a, const int n, const double& v);
void arraySetAll(float* d_a, const int n, const float& v);
void arraySetAll(int* d_a, const int n, const int& v);
void arraySetAll(doubleComplex* d_a, const int n, const doubleComplex& v);
void arraySetAll(floatComplex* d_a, const int n, const floatComplex& v);

void arrayScaleAll(double* d_a, const int n, const double& v);
void arrayScaleAll(float* d_a, const int n, const float& v);
void arrayScaleAll(int* d_a, const int n, const int& v);
void arrayScaleAll(doubleComplex* d_a, const int n, const doubleComplex& v);
void arrayScaleAll(floatComplex* d_a, const int n, const floatComplex& v);

void arrayMultAll(double* d_dest, double* d_src1, double* d_src2, const int n);
void arrayMultAll(float* d_dest, float* d_src1, float* d_src2, const int n);
void arrayMultAll(int* d_dest, int* d_src1, int* d_src2, const int n);
void arrayMultAll(doubleComplex* d_dest, doubleComplex* d_src1, doubleComplex* d_src2, const int n);
void arrayMultAll(floatComplex* d_dest, floatComplex* d_src1, floatComplex* d_src2, const int n);

void arrayDiffAll(double* d_dest, double* d_src1, double* d_src2, const int n);
void arrayDiffAll(float* d_dest, float* d_src1, float* d_src2, const int n);
void arrayDiffAll(int* d_dest, int* d_src1, int* d_src2, const int n);
void arrayDiffAll(doubleComplex* d_dest, doubleComplex* d_src1, doubleComplex* d_src2, const int n);
void arrayDiffAll(floatComplex* d_dest, floatComplex* d_src1, floatComplex* d_src2, const int n);

void arrayNormAll(double* d_dest, double* d_src1, const int n);
void arrayNormAll(float* d_dest, float* d_src1, const int n);
void arrayNormAll(int* d_dest, int* d_src1, const int n);
void arrayNormAll(doubleComplex* d_dest, doubleComplex* d_src1, const int n);
void arrayNormAll(floatComplex* d_dest, floatComplex* d_src1, const int n);

void arraySumAll(double* d_a, const int n, double& v);
void arraySumAll(float* d_a, const int n, float& v);
void arraySumAll(int* d_a, const int n, int& v);
void arraySumAll(doubleComplex* d_a, const int n, doubleComplex& v);
void arraySumAll(floatComplex* d_a, const int n, floatComplex& v);

void arrayGetRealPart(double* d_dest, const doubleComplex* d_src, const int n);
void arrayGetRealPart(float* d_dest, const floatComplex* d_src, const int n);

void arrayGetImagPart(double* d_dest, const doubleComplex* d_src, const int n);
void arrayGetImagPart(float* d_dest, const floatComplex* d_src, const int n);

void arraySetRealPart(doubleComplex* d_dest, const double * d_src, const int n);
void arraySetRealPart(floatComplex* d_dest, const float * d_src, const int n);

void arraySetImagPart(doubleComplex* d_dest, const double * d_src, const int n);
void arraySetImagPart(floatComplex* d_dest, const float * d_src, const int n);



#endif
