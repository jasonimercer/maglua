#include "array_ops.hpp" //typedefs

typedef struct FFT_PLAN FFT_PLAN;
#define FFT_FORWARD -1
#define FFT_BACKWARD 1



FFT_PLAN* make_FFT_PLAN_double(int direction, int fftdims, const int nx, const int ny, const int nz);
FFT_PLAN* make_FFT_PLAN_float(int direction, int fftdims, const int nx, const int ny, const int nz);

template<typename T>
inline FFT_PLAN* make_FFT_PLAN_T(int direction, int fftdims, const int nx, const int ny, const int nz)
{
// 	fprintf(stderr, "this shouldn't be happening (%s:%i)\n", __FILE__, __LINE__);
	fprintf(stderr, "FFTs not supported for this datatype (try a complex datatype).\n");
	return 0;
}
template<>
inline FFT_PLAN* make_FFT_PLAN_T<doubleComplex>(int direction, int fftdims, const int nx, const int ny, const int nz)
{
	return make_FFT_PLAN_double(direction, fftdims, nx, ny, nz);
}
template<>
inline FFT_PLAN* make_FFT_PLAN_T<floatComplex>(int direction, int fftdims, const int nx, const int ny, const int nz)
{
	return make_FFT_PLAN_float(direction, fftdims, nx, ny, nz);
}


void execute_FFT_PLAN(FFT_PLAN* plan, floatComplex* dest, floatComplex* src, floatComplex* ws);
void execute_FFT_PLAN(FFT_PLAN* plan, doubleComplex* dest, doubleComplex* src, doubleComplex* ws);





void free_FFT_PLAN(FFT_PLAN* p);
