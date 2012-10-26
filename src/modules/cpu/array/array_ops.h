#ifdef WIN32
 #ifdef ARRAY_EXPORTS
  #define ARRAY_API __declspec(dllexport)
 #else
  #define ARRAY_API __declspec(dllimport)
 #endif
#else
 #define ARRAY_API 
#endif

#include <complex>
#ifndef COMPLEX_TYPES
#define COMPLEX_TYPES
using namespace std;
typedef complex<double> doubleComplex; //cpu version
typedef complex<float>   floatComplex;
#endif

#ifdef WIN32
	#define DOUBLE_ARRAY
	#define SINGLE_ARRAY
#else
	#define DOUBLE_ARRAY
	#define SINGLE_ARRAY
#endif

#ifndef ARRAY_SUPPORT_FUNCTIONS
#define ARRAY_SUPPORT_FUNCTIONS


ARRAY_API void arraySetAll(double* d_a, const double& v, const int n);
ARRAY_API void arraySetAll(float* d_a,  const float& v, const int n);
ARRAY_API void arraySetAll(int* d_a, const int& v, const int n);
ARRAY_API void arraySetAll(doubleComplex* d_a, const doubleComplex& v, const int n);
ARRAY_API void arraySetAll(floatComplex* d_a,  const floatComplex& v, const int n);

ARRAY_API void arrayScaleAll(double* d_a, const double& v, const int n);
ARRAY_API void arrayScaleAll(float* d_a, const float& v, const int n);
ARRAY_API void arrayScaleAll(int* d_a, const int& v, const int n);
ARRAY_API void arrayScaleAll(doubleComplex* d_a, const doubleComplex& v, const int n);
ARRAY_API void arrayScaleAll(floatComplex* d_a, const floatComplex& v, const int n);


ARRAY_API void arrayScaleAll_o(double* d_a, const int offset, const double& v, const int n);
ARRAY_API void arrayScaleAll_o(float* d_a, const int offset, const float& v, const int n);
ARRAY_API void arrayScaleAll_o(int* d_a, const int offset, const int& v, const int n);
ARRAY_API void arrayScaleAll_o(doubleComplex* d_a, const int offset, const doubleComplex& v, const int n);
ARRAY_API void arrayScaleAll_o(floatComplex* d_a, const int offset, const floatComplex& v, const int n);

ARRAY_API void arrayAddAll(double* d_a, const double& v, const int n);
ARRAY_API void arrayAddAll(float* d_a, const float& v, const int n);
ARRAY_API void arrayAddAll(int* d_a, const int& v, const int n);
ARRAY_API void arrayAddAll(doubleComplex* d_a, const doubleComplex& v, const int n);
ARRAY_API void arrayAddAll(floatComplex* d_a, const floatComplex& v, const int n);

ARRAY_API void arrayMultAll(double* d_dest, double* d_src1, double* d_src2, const int n);
ARRAY_API void arrayMultAll(float* d_dest, float* d_src1, float* d_src2, const int n);
ARRAY_API void arrayMultAll(int* d_dest, int* d_src1, int* d_src2, const int n);
ARRAY_API void arrayMultAll(doubleComplex* d_dest, doubleComplex* d_src1, doubleComplex* d_src2, const int n);
ARRAY_API void arrayMultAll(floatComplex* d_dest, floatComplex* d_src1, floatComplex* d_src2, const int n);

ARRAY_API void arrayDiffAll(double* d_dest, double* d_src1, double* d_src2, const int n);
ARRAY_API void arrayDiffAll(float* d_dest, float* d_src1, float* d_src2, const int n);
ARRAY_API void arrayDiffAll(int* d_dest, int* d_src1, int* d_src2, const int n);
ARRAY_API void arrayDiffAll(doubleComplex* d_dest, doubleComplex* d_src1, doubleComplex* d_src2, const int n);
ARRAY_API void arrayDiffAll(floatComplex* d_dest, floatComplex* d_src1, floatComplex* d_src2, const int n);

ARRAY_API void arraySumAll(double* d_dest, const double* d_src1, const double* d_src2, const int n);
ARRAY_API void arraySumAll( float* d_dest,  const float* d_src1,  const float* d_src2, const int n);
ARRAY_API void arraySumAll(   int* d_dest,    const int* d_src1,    const int* d_src2, const int n);
ARRAY_API void arraySumAll(doubleComplex* d_dest, const doubleComplex* d_src1, const doubleComplex* d_src2, const int n);
ARRAY_API void arraySumAll( floatComplex* d_dest,  const floatComplex* d_src1,  const floatComplex* d_src2, const int n);

ARRAY_API void reduceSumAll(const double* d, const int n, double& v);
ARRAY_API void reduceSumAll(const  float* d, const int n, float& v);
ARRAY_API void reduceSumAll(const    int* d, const int n, int& v);
ARRAY_API void reduceSumAll(const doubleComplex* d, const int n, doubleComplex& v);
ARRAY_API void reduceSumAll(const  floatComplex* d, const int n, floatComplex& v);

ARRAY_API void arrayNormAll(double* d_dest, double* d_src1, const int n);
ARRAY_API void arrayNormAll(float* d_dest, float* d_src1, const int n);
ARRAY_API void arrayNormAll(int* d_dest, int* d_src1, const int n);
ARRAY_API void arrayNormAll(doubleComplex* d_dest, doubleComplex* d_src1, const int n);
ARRAY_API void arrayNormAll(floatComplex* d_dest, floatComplex* d_src1, const int n);

ARRAY_API void reduceDiffSumAll(const double* d_a, const double* d_b, const int n, double& v);
ARRAY_API void reduceDiffSumAll(const float* d_a, const float* d_b, const int n, float& v);
ARRAY_API void reduceDiffSumAll(const int* d_a, const int* d_b, const int n, int& v);
ARRAY_API void reduceDiffSumAll(const doubleComplex* d_a, const doubleComplex* d_b, const int n, doubleComplex& v);
ARRAY_API void reduceDiffSumAll(const floatComplex* d_a, const floatComplex* d_b, const int n, floatComplex& v);


ARRAY_API void reduceMultSumAll(const double* d_a, const double* d_b, const int n, double& v);
ARRAY_API void reduceMultSumAll(const float* d_a, const float* d_b, const int n, float& v);
ARRAY_API void reduceMultSumAll(const int* d_a, const int* d_b, const int n, int& v);
ARRAY_API void reduceMultSumAll(const doubleComplex* d_a, const doubleComplex* d_b, const int n, doubleComplex& v);
ARRAY_API void reduceMultSumAll(const floatComplex* d_a, const floatComplex* d_b, const int n, floatComplex& v);


ARRAY_API void reduceExtreme(const double* d_a, const int min_max, const int n, double& v, int& idx);
ARRAY_API void reduceExtreme(const float* d_a, const int min_max, const int n, float& v, int& idx);
ARRAY_API void reduceExtreme(const int* d_a, const int min_max, const int n, int& v, int& idx);
ARRAY_API void reduceExtreme(const doubleComplex* d_a, const int min_max, const int n, doubleComplex& v, int& idx);
ARRAY_API void reduceExtreme(const floatComplex* d_a, const int min_max, const int n, floatComplex& v, int& idx);

ARRAY_API void arrayScaleAdd(double* dest, double s1, const double* src1, const double s2, const double* src2, const int n);
ARRAY_API void arrayScaleAdd(float* dest, float s1, const float* src1, float s2, const float* src2, const int n);
ARRAY_API void arrayScaleAdd(int* dest, int s1, const int* src1, int s2, const int* src2, const int n);
ARRAY_API void arrayScaleAdd(doubleComplex* dest, doubleComplex s1, const doubleComplex* src1, doubleComplex s2, const doubleComplex* src2, const int n);
ARRAY_API void arrayScaleAdd(floatComplex* dest, floatComplex s1, const floatComplex* src1, floatComplex s2, const floatComplex* src2, const int n);


ARRAY_API void arrayGetRealPart(double* d_dest, const doubleComplex* d_src, const int n);
ARRAY_API void arrayGetRealPart(float* d_dest, const floatComplex* d_src, const int n);

ARRAY_API void arrayGetImagPart(double* d_dest, const doubleComplex* d_src, const int n);
ARRAY_API void arrayGetImagPart(float* d_dest, const floatComplex* d_src, const int n);

ARRAY_API void arraySetRealPart(doubleComplex* d_dest, const double * d_src, const int n);
ARRAY_API void arraySetRealPart(floatComplex* d_dest, const float * d_src, const int n);

ARRAY_API void arraySetImagPart(doubleComplex* d_dest, const double * d_src, const int n);
ARRAY_API void arraySetImagPart(floatComplex* d_dest, const float * d_src, const int n);


ARRAY_API void arrayLayerMult(double* dest, int dest_layer, const double* src1, int src1_layer, const double* src2, int src2_layer, double mult, int set, const int nxy);
ARRAY_API void arrayLayerMult(float* dest, int dest_layer, const float* src1, int src1_layer, const float* src2, int src2_layer, float mult, int set, const int nxy);
ARRAY_API void arrayLayerMult(int* dest, int dest_layer, const int* src1, int src1_layer, const int* src2, int src2_layer, int mult, int set, const int nxy);
ARRAY_API void arrayLayerMult(doubleComplex* dest, int dest_layer, const doubleComplex* src1, int src1_layer, const doubleComplex* src2, int src2_layer, doubleComplex mult, int set, const int nxy);
ARRAY_API void arrayLayerMult(floatComplex* dest, int dest_layer, const floatComplex* src1, int src1_layer, const floatComplex* src2, int src2_layer, floatComplex mult, int set, const int nxy);

// _o arbitrarily means offset
ARRAY_API void arrayScaleMultAdd_o(double* dest, const int od, double scale, const double* src1, const int o1, const double* src2, const int o2, const double* src3, const int o3, const int nxy);
ARRAY_API void arrayScaleMultAdd_o(float* dest, const int od, float scale, const float* src1, const int o1, const float* src2, const int o2, const float* src3, const int o3, const int nxy);
ARRAY_API void arrayScaleMultAdd_o(int* dest, const int od, int scale, const int* src1, const int o1, const int* src2, const int o2, const int* src3, const int o3, const int nxy);
ARRAY_API void arrayScaleMultAdd_o(doubleComplex* dest, const int od, doubleComplex scale, const doubleComplex* src1, const int o1, const doubleComplex* src2, const int o2, const doubleComplex* src3, const int o3, const int nxy);
ARRAY_API void arrayScaleMultAdd_o(floatComplex* dest, const int od, floatComplex scale, const floatComplex* src1, const floatComplex* src2, const int o2, const floatComplex* src3, const int o3, const int nxy);



#endif
