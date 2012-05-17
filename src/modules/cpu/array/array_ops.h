#include <complex>
#ifndef COMPLEX_TYPES
#define COMPLEX_TYPES
using namespace std;
typedef complex<double> doubleComplex; //cpu version
typedef complex<float>   floatComplex;
#endif

#ifndef ARRAY_SUPPORT_FUNCTIONS
#define ARRAY_SUPPORT_FUNCTIONS


void arraySetAll(double* d_a, const double& v, const int n);
void arraySetAll(float* d_a,  const float& v, const int n);
void arraySetAll(int* d_a, const int& v, const int n);
void arraySetAll(doubleComplex* d_a, const doubleComplex& v, const int n);
void arraySetAll(floatComplex* d_a,  const floatComplex& v, const int n);

void arrayScaleAll(double* d_a, const double& v, const int n);
void arrayScaleAll(float* d_a, const float& v, const int n);
void arrayScaleAll(int* d_a, const int& v, const int n);
void arrayScaleAll(doubleComplex* d_a, const doubleComplex& v, const int n);
void arrayScaleAll(floatComplex* d_a, const floatComplex& v, const int n);


void arrayAddAll(double* d_a, const double& v, const int n);
void arrayAddAll(float* d_a, const float& v, const int n);
void arrayAddAll(int* d_a, const int& v, const int n);
void arrayAddAll(doubleComplex* d_a, const doubleComplex& v, const int n);
void arrayAddAll(floatComplex* d_a, const floatComplex& v, const int n);

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

void arraySumAll(double* d_dest, const double* d_src1, const double* d_src2, const int n);
void arraySumAll( float* d_dest,  const float* d_src1,  const float* d_src2, const int n);
void arraySumAll(   int* d_dest,    const int* d_src1,    const int* d_src2, const int n);
void arraySumAll(doubleComplex* d_dest, const doubleComplex* d_src1, const doubleComplex* d_src2, const int n);
void arraySumAll( floatComplex* d_dest,  const floatComplex* d_src1,  const floatComplex* d_src2, const int n);

void reduceSumAll(const double* d, const int n, double& v);
void reduceSumAll(const  float* d, const int n, float& v);
void reduceSumAll(const    int* d, const int n, int& v);
void reduceSumAll(const doubleComplex* d, const int n, doubleComplex& v);
void reduceSumAll(const  floatComplex* d, const int n, floatComplex& v);

void arrayNormAll(double* d_dest, double* d_src1, const int n);
void arrayNormAll(float* d_dest, float* d_src1, const int n);
void arrayNormAll(int* d_dest, int* d_src1, const int n);
void arrayNormAll(doubleComplex* d_dest, doubleComplex* d_src1, const int n);
void arrayNormAll(floatComplex* d_dest, floatComplex* d_src1, const int n);

void reduceDiffSumAll(const double* d_a, const double* d_b, const int n, double& v);
void reduceDiffSumAll(const float* d_a, const float* d_b, const int n, float& v);
void reduceDiffSumAll(const int* d_a, const int* d_b, const int n, int& v);
void reduceDiffSumAll(const doubleComplex* d_a, const doubleComplex* d_b, const int n, doubleComplex& v);
void reduceDiffSumAll(const floatComplex* d_a, const floatComplex* d_b, const int n, floatComplex& v);

void arrayScaleAdd(double* dest, double s1, const double* src1, const double s2, const double* src2, const int n);
void arrayScaleAdd(float* dest, float s1, const float* src1, float s2, const float* src2, const int n);
void arrayScaleAdd(int* dest, int s1, const int* src1, int s2, const int* src2, const int n);
void arrayScaleAdd(doubleComplex* dest, doubleComplex s1, const doubleComplex* src1, doubleComplex s2, const doubleComplex* src2, const int n);
void arrayScaleAdd(floatComplex* dest, floatComplex s1, const floatComplex* src1, floatComplex s2, const floatComplex* src2, const int n);


void arrayGetRealPart(double* d_dest, const doubleComplex* d_src, const int n);
void arrayGetRealPart(float* d_dest, const floatComplex* d_src, const int n);

void arrayGetImagPart(double* d_dest, const doubleComplex* d_src, const int n);
void arrayGetImagPart(float* d_dest, const floatComplex* d_src, const int n);

void arraySetRealPart(doubleComplex* d_dest, const double * d_src, const int n);
void arraySetRealPart(floatComplex* d_dest, const float * d_src, const int n);

void arraySetImagPart(doubleComplex* d_dest, const double * d_src, const int n);
void arraySetImagPart(floatComplex* d_dest, const float * d_src, const int n);




void arrayLayerMult(double* dest, int dest_layer, const double* src1, int src1_layer, const double* src2, int src2_layer, double mult, int set, const int nxy);
void arrayLayerMult(float* dest, int dest_layer, const float* src1, int src1_layer, const float* src2, int src2_layer, float mult, int set, const int nxy);
void arrayLayerMult(int* dest, int dest_layer, const int* src1, int src1_layer, const int* src2, int src2_layer, int mult, int set, const int nxy);
void arrayLayerMult(doubleComplex* dest, int dest_layer, const doubleComplex* src1, int src1_layer, const doubleComplex* src2, int src2_layer, doubleComplex mult, int set, const int nxy);
void arrayLayerMult(floatComplex* dest, int dest_layer, const floatComplex* src1, int src1_layer, const floatComplex* src2, int src2_layer, floatComplex mult, int set, const int nxy);



#endif
