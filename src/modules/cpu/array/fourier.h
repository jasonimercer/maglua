#include "array_ops.h" //typedefs
#include <stdio.h>

#ifndef FOURIER_FUNCTIONS_ARRAY
#define FOURIER_FUNCTIONS_ARRAY

typedef struct FFT_PLAN FFT_PLAN;
#define FFT_FORWARD -1
#define FFT_BACKWARD 1


#ifdef DOUBLE_ARRAY
ARRAY_API FFT_PLAN* make_FFT_PLAN_double(int direction, int fftdims, const int nx, const int ny, const int nz);
#endif

#ifdef SINGLE_ARRAY
ARRAY_API FFT_PLAN* make_FFT_PLAN_float(int direction, int fftdims, const int nx, const int ny, const int nz);
#endif

template<typename T>
ARRAY_API inline FFT_PLAN* make_FFT_PLAN_T(int direction, int fftdims, const int nx, const int ny, const int nz)
{
// 	fprintf(stderr, "this shouldn't be happening (%s:%i)\n", __FILE__, __LINE__);
	fprintf(stderr, "FFTs not supported for this datatype (try a complex datatype).\n");
	return 0;
}
#ifdef DOUBLE_ARRAY
template<>
ARRAY_API inline FFT_PLAN* make_FFT_PLAN_T<doubleComplex>(int direction, int fftdims, const int nx, const int ny, const int nz)
{
	return make_FFT_PLAN_double(direction, fftdims, nx, ny, nz);
}
#endif
#ifdef SINGLE_ARRAY
template<>
ARRAY_API inline FFT_PLAN* make_FFT_PLAN_T<floatComplex>(int direction, int fftdims, const int nx, const int ny, const int nz)
{
	return make_FFT_PLAN_float(direction, fftdims, nx, ny, nz);
}
#endif


#ifdef DOUBLE_ARRAY
ARRAY_API void execute_FFT_PLAN(FFT_PLAN* plan, floatComplex* dest, floatComplex* src, floatComplex* ws);
#endif
#ifdef SINGLE_ARRAY
ARRAY_API void execute_FFT_PLAN(FFT_PLAN* plan, doubleComplex* dest, doubleComplex* src, doubleComplex* ws);
#endif

#ifdef WIN32
// BS functions so that windows template exports are happy
ARRAY_API void execute_FFT_PLAN(FFT_PLAN* plan, int* dest, int* src, int* ws);
ARRAY_API void execute_FFT_PLAN(FFT_PLAN* plan, float* dest, float* src, float* ws);
ARRAY_API void execute_FFT_PLAN(FFT_PLAN* plan, double* dest, double* src, double* ws);
#endif


void free_FFT_PLAN(FFT_PLAN* p);

#endif

