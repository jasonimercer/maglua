#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

#ifndef CUDA_COMPLEX_TYPES
#define CUDA_COMPLEX_TYPES
typedef cuDoubleComplex doubleComplex; //cuda version
typedef cuFloatComplex floatComplex;
#endif


#ifdef WIN32
 #ifdef ARRAY_CUDA_EXPORTS
  #define ARRAY_CUDA_API __declspec(dllexport)
 #else
  #define ARRAY_CUDA_API __declspec(dllimport)
 #endif
#else
 #define ARRAY_CUDA_API
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


ARRAY_CUDA_API void arraySetAll(double* d_a, const double& v, const int n);
ARRAY_CUDA_API void arraySetAll(float* d_a,  const float& v, const int n);
ARRAY_CUDA_API void arraySetAll(int* d_a, const int& v, const int n);
ARRAY_CUDA_API void arraySetAll(doubleComplex* d_a, const doubleComplex& v, const int n);
ARRAY_CUDA_API void arraySetAll(floatComplex* d_a,  const floatComplex& v, const int n);

ARRAY_CUDA_API void arrayScaleAll(double* d_a, const double& v, const int n);
ARRAY_CUDA_API void arrayScaleAll(float* d_a, const float& v, const int n);
ARRAY_CUDA_API void arrayScaleAll(int* d_a, const int& v, const int n);
ARRAY_CUDA_API void arrayScaleAll(doubleComplex* d_a, const doubleComplex& v, const int n);
ARRAY_CUDA_API void arrayScaleAll(floatComplex* d_a, const floatComplex& v, const int n);


ARRAY_CUDA_API void arrayScaleAll_o(double* d_a, const int offset, const double& v, const int n);
ARRAY_CUDA_API void arrayScaleAll_o(float* d_a, const int offset, const float& v, const int n);
ARRAY_CUDA_API void arrayScaleAll_o(int* d_a, const int offset, const int& v, const int n);
ARRAY_CUDA_API void arrayScaleAll_o(doubleComplex* d_a, const int offset, const doubleComplex& v, const int n);
ARRAY_CUDA_API void arrayScaleAll_o(floatComplex* d_a, const int offset, const floatComplex& v, const int n);

ARRAY_CUDA_API void arrayAddAll(double* d_a, const double& v, const int n);
ARRAY_CUDA_API void arrayAddAll(float* d_a, const float& v, const int n);
ARRAY_CUDA_API void arrayAddAll(int* d_a, const int& v, const int n);
ARRAY_CUDA_API void arrayAddAll(doubleComplex* d_a, const doubleComplex& v, const int n);
ARRAY_CUDA_API void arrayAddAll(floatComplex* d_a, const floatComplex& v, const int n);

ARRAY_CUDA_API void arrayMultAll(double* d_dest, double* d_src1, double* d_src2, const int n);
ARRAY_CUDA_API void arrayMultAll(float* d_dest, float* d_src1, float* d_src2, const int n);
ARRAY_CUDA_API void arrayMultAll(int* d_dest, int* d_src1, int* d_src2, const int n);
ARRAY_CUDA_API void arrayMultAll(doubleComplex* d_dest, doubleComplex* d_src1, doubleComplex* d_src2, const int n);
ARRAY_CUDA_API void arrayMultAll(floatComplex* d_dest, floatComplex* d_src1, floatComplex* d_src2, const int n);

ARRAY_CUDA_API void arrayDiffAll(double* d_dest, double* d_src1, double* d_src2, const int n);
ARRAY_CUDA_API void arrayDiffAll(float* d_dest, float* d_src1, float* d_src2, const int n);
ARRAY_CUDA_API void arrayDiffAll(int* d_dest, int* d_src1, int* d_src2, const int n);
ARRAY_CUDA_API void arrayDiffAll(doubleComplex* d_dest, doubleComplex* d_src1, doubleComplex* d_src2, const int n);
ARRAY_CUDA_API void arrayDiffAll(floatComplex* d_dest, floatComplex* d_src1, floatComplex* d_src2, const int n);

ARRAY_CUDA_API void arraySumAll(double* d_dest, const double* d_src1, const double* d_src2, const int n);
ARRAY_CUDA_API void arraySumAll( float* d_dest,  const float* d_src1,  const float* d_src2, const int n);
ARRAY_CUDA_API void arraySumAll(   int* d_dest,    const int* d_src1,    const int* d_src2, const int n);
ARRAY_CUDA_API void arraySumAll(doubleComplex* d_dest, const doubleComplex* d_src1, const doubleComplex* d_src2, const int n);
ARRAY_CUDA_API void arraySumAll( floatComplex* d_dest,  const floatComplex* d_src1,  const floatComplex* d_src2, const int n);

ARRAY_CUDA_API void reducePowerSumAll(const double* d_a, double p, const int n, double& v);
ARRAY_CUDA_API void reducePowerSumAll(const  float* d_a, double p, const int n, float& v);
ARRAY_CUDA_API void reducePowerSumAll(const    int* d_a, double p, const int n, int& v);
ARRAY_CUDA_API void reducePowerSumAll(const doubleComplex* d_a, double p, const int n, doubleComplex& v);
ARRAY_CUDA_API void reducePowerSumAll(const floatComplex* d_a, double p, const int n, floatComplex& v);


ARRAY_CUDA_API void arrayNormAll(double* d_dest, double* d_src1, const int n);
ARRAY_CUDA_API void arrayNormAll(float* d_dest, float* d_src1, const int n);
ARRAY_CUDA_API void arrayNormAll(int* d_dest, int* d_src1, const int n);
ARRAY_CUDA_API void arrayNormAll(doubleComplex* d_dest, doubleComplex* d_src1, const int n);
ARRAY_CUDA_API void arrayNormAll(floatComplex* d_dest, floatComplex* d_src1, const int n);

ARRAY_CUDA_API void reduceDiffSumAll(const double* d_a, const double* d_b, const int n, double& v);
ARRAY_CUDA_API void reduceDiffSumAll(const float* d_a, const float* d_b, const int n, float& v);
ARRAY_CUDA_API void reduceDiffSumAll(const int* d_a, const int* d_b, const int n, int& v);
ARRAY_CUDA_API void reduceDiffSumAll(const doubleComplex* d_a, const doubleComplex* d_b, const int n, doubleComplex& v);
ARRAY_CUDA_API void reduceDiffSumAll(const floatComplex* d_a, const floatComplex* d_b, const int n, floatComplex& v);


ARRAY_CUDA_API void reduceMultSumAll(const double* d_a, const double* d_b, const int n, double& v);
ARRAY_CUDA_API void reduceMultSumAll(const float* d_a, const float* d_b, const int n, float& v);
ARRAY_CUDA_API void reduceMultSumAll(const int* d_a, const int* d_b, const int n, int& v);
ARRAY_CUDA_API void reduceMultSumAll(const doubleComplex* d_a, const doubleComplex* d_b, const int n, doubleComplex& v);
ARRAY_CUDA_API void reduceMultSumAll(const floatComplex* d_a, const floatComplex* d_b, const int n, floatComplex& v);


ARRAY_CUDA_API void reduceExtreme(const double* d_a, const int min_max, const int n, double& v, int& idx);
ARRAY_CUDA_API void reduceExtreme(const float* d_a, const int min_max, const int n, float& v, int& idx);
ARRAY_CUDA_API void reduceExtreme(const int* d_a, const int min_max, const int n, int& v, int& idx);
ARRAY_CUDA_API void reduceExtreme(const doubleComplex* d_a, const int min_max, const int n, doubleComplex& v, int& idx);
ARRAY_CUDA_API void reduceExtreme(const floatComplex* d_a, const int min_max, const int n, floatComplex& v, int& idx);

ARRAY_CUDA_API void arrayScaleAdd(double* dest, double s1, const double* src1, const double s2, const double* src2, const int n);
ARRAY_CUDA_API void arrayScaleAdd(float* dest, float s1, const float* src1, float s2, const float* src2, const int n);
ARRAY_CUDA_API void arrayScaleAdd(int* dest, int s1, const int* src1, int s2, const int* src2, const int n);
ARRAY_CUDA_API void arrayScaleAdd(doubleComplex* dest, doubleComplex s1, const doubleComplex* src1, doubleComplex s2, const doubleComplex* src2, const int n);
ARRAY_CUDA_API void arrayScaleAdd(floatComplex* dest, floatComplex s1, const floatComplex* src1, floatComplex s2, const floatComplex* src2, const int n);


ARRAY_CUDA_API void arrayGetRealPart(double* d_dest, const doubleComplex* d_src, const int n);
ARRAY_CUDA_API void arrayGetRealPart(float* d_dest, const floatComplex* d_src, const int n);

ARRAY_CUDA_API void arrayGetImagPart(double* d_dest, const doubleComplex* d_src, const int n);
ARRAY_CUDA_API void arrayGetImagPart(float* d_dest, const floatComplex* d_src, const int n);

ARRAY_CUDA_API void arraySetRealPart(doubleComplex* d_dest, const double * d_src, const int n);
ARRAY_CUDA_API void arraySetRealPart(floatComplex* d_dest, const float * d_src, const int n);

ARRAY_CUDA_API void arraySetImagPart(doubleComplex* d_dest, const double * d_src, const int n);
ARRAY_CUDA_API void arraySetImagPart(floatComplex* d_dest, const float * d_src, const int n);


ARRAY_CUDA_API void arrayLayerMult(double* dest, int dest_layer, const double* src1, int src1_layer, const double* src2, int src2_layer, double mult, int set, const int nxy);
ARRAY_CUDA_API void arrayLayerMult(float* dest, int dest_layer, const float* src1, int src1_layer, const float* src2, int src2_layer, float mult, int set, const int nxy);
ARRAY_CUDA_API void arrayLayerMult(int* dest, int dest_layer, const int* src1, int src1_layer, const int* src2, int src2_layer, int mult, int set, const int nxy);
ARRAY_CUDA_API void arrayLayerMult(doubleComplex* dest, int dest_layer, const doubleComplex* src1, int src1_layer, const doubleComplex* src2, int src2_layer, doubleComplex mult, int set, const int nxy);
ARRAY_CUDA_API void arrayLayerMult(floatComplex* dest, int dest_layer, const floatComplex* src1, int src1_layer, const floatComplex* src2, int src2_layer, floatComplex mult, int set, const int nxy);

// _o arbitrarily means offset
ARRAY_CUDA_API void arrayScaleMultAdd_o(double* dest, const int od, double scale, const double* src1, const int o1, const double* src2, const int o2, const double* src3, const int o3, const int nxy);
ARRAY_CUDA_API void arrayScaleMultAdd_o(float* dest, const int od, float scale, const float* src1, const int o1, const float* src2, const int o2, const float* src3, const int o3, const int nxy);
ARRAY_CUDA_API void arrayScaleMultAdd_o(int* dest, const int od, int scale, const int* src1, const int o1, const int* src2, const int o2, const int* src3, const int o3, const int nxy);
ARRAY_CUDA_API void arrayScaleMultAdd_o(doubleComplex* dest, const int od, doubleComplex scale, const doubleComplex* src1, const int o1, const doubleComplex* src2, const int o2, const doubleComplex* src3, const int o3, const int nxy);
ARRAY_CUDA_API void arrayScaleMultAdd_o(floatComplex* dest, const int od, floatComplex scale, const floatComplex* src1, const floatComplex* src2, const int o2, const floatComplex* src3, const int o3, const int nxy);

ARRAY_CUDA_API void arrayCopyRegionFromTo(double* src, int sx, int sy, int sz, int*s1,int*s2,double* dest, int dx, int dy, int dz, int* d1, int* d2);
ARRAY_CUDA_API void arrayCopyRegionFromTo( float* src, int sx, int sy, int sz, int*s1,int*s2, float* dest, int dx, int dy, int dz, int* d1, int* d2);
ARRAY_CUDA_API void arrayCopyRegionFromTo(   int* src, int sx, int sy, int sz, int*s1,int*s2,   int* dest, int dx, int dy, int dz, int* d1, int* d2);
ARRAY_CUDA_API void arrayCopyRegionFromTo(doubleComplex* src, int sx, int sy, int sz, int*s1,int*s2,doubleComplex* dest, int dx, int dy, int dz, int* d1, int* d2);
ARRAY_CUDA_API void arrayCopyRegionFromTo( floatComplex* src, int sx, int sy, int sz, int*s1,int*s2, floatComplex* dest, int dx, int dy, int dz, int* d1, int* d2);


bool arrayAreAllSameValue(double* d_data, const int n, double& v);
bool arrayAreAllSameValue(float* d_data, const int n, float& v);
bool arrayAreAllSameValue(int* d_data, const int n, int& v);
bool arrayAreAllSameValue(doubleComplex* d_data, const int n, doubleComplex& v);
bool arrayAreAllSameValue(floatComplex* d_data, const int n, floatComplex& v);

#endif //ARRAY_SUPPORT_FUNCTIONS



//orig
#if 0


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

void arraySumAll(double* d_dest, const double* d_src1, const double* d_src2, const int n);
void arraySumAll( float* d_dest,  const float* d_src1,  const float* d_src2, const int n);
void arraySumAll(   int* d_dest,    const int* d_src1,    const int* d_src2, const int n);
void arraySumAll(doubleComplex* d_dest, const doubleComplex* d_src1, const doubleComplex* d_src2, const int n);
void arraySumAll( floatComplex* d_dest,  const floatComplex* d_src1,  const floatComplex* d_src2, const int n);

// add a single value to all elements
void arrayAddAll(double* d_a, const double& v, const int n);
void arrayAddAll(float* d_a, const float& v, const int n);
void arrayAddAll(int* d_a, const int& v, const int n);
void arrayAddAll(doubleComplex* d_a, const doubleComplex& v, const int n);
void arrayAddAll(floatComplex* d_a, const floatComplex& v, const int n);

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

void arrayNormAll(double* d_dest, const double* d_src1, const int n);
void arrayNormAll(float* d_dest, const float* d_src1, const int n);
void arrayNormAll(int* d_dest, const int* d_src1, const int n);
void arrayNormAll(doubleComplex* d_dest, const doubleComplex* d_src1, const int n);
void arrayNormAll(floatComplex* d_dest, const floatComplex* d_src1, const int n);


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

bool arrayAreAllSameValue(double* d_data, const int n, double& v);
bool arrayAreAllSameValue(float* d_data, const int n, float& v);
bool arrayAreAllSameValue(int* d_data, const int n, int& v);
bool arrayAreAllSameValue(doubleComplex* d_data, const int n, doubleComplex& v);
bool arrayAreAllSameValue(floatComplex* d_data, const int n, floatComplex& v);
#endif


