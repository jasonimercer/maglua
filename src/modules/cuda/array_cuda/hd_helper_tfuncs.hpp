#include <cuComplex.h>
typedef cuDoubleComplex doubleComplex; //cuda version
typedef cuFloatComplex floatComplex;


template<typename T>
__host__ __device__ inline void plus_equal(T& lhs, const T& rhs)
{
	lhs += rhs;
}
template<>
__host__ __device__ inline void plus_equal<floatComplex>(floatComplex& lhs, const floatComplex& rhs)
{
	lhs = cuCaddf(lhs, rhs);
}
template<>
__host__ __device__ inline void plus_equal<doubleComplex>(doubleComplex& lhs, const doubleComplex& rhs)
{
	lhs = cuCadd(lhs, rhs);
}


template<typename T>
__host__ __device__ inline void times_equal(T& lhs, const T& rhs)
{
	lhs *= rhs;
}
template<>
__host__ __device__ inline void times_equal<floatComplex>(floatComplex& lhs, const floatComplex& rhs)
{
	lhs = cuCmulf(lhs, rhs);
}
template<>
__device__ inline void times_equal<doubleComplex>(doubleComplex& lhs, const doubleComplex& rhs)
{
	lhs = cuCmul(lhs, rhs);
}


template<typename T>
__host__ __device__ inline T zero()
{
	return 0;
}
template<>
__host__ __device__ inline floatComplex zero<floatComplex>()
{
	return make_floatComplex(0,0);
}
template<>
__host__ __device__ inline doubleComplex zero<doubleComplex>()
{
	return make_doubleComplex(0,0);
}


template<typename T>
__host__ __device__ inline T one()
{
	return 1;
}
template<>
__host__ __device__ inline floatComplex one<floatComplex>()
{
	return make_floatComplex(1,0);
}
template<>
__host__ __device__ inline doubleComplex one<doubleComplex>()
{
	return make_doubleComplex(1,0);
}




template<typename T>
__host__ __device__ inline void set_norm(T& dest, const T& src)
{
	dest = 0;
}
template<>
__host__ __device__ inline void set_norm<int>(int& dest, const int& src) 
{
	dest = abs(src);
}
template<>
__host__ __device__ inline void set_norm<float>(float& dest, const float& src) 
{
	dest = fabsf(src);
}
template<>
__host__ __device__ inline void set_norm<double>(double& dest, const double& src) 
{
	dest = fabs(src);
}
template<>
__host__ __device__ inline void set_norm<floatComplex>(floatComplex& dest, const floatComplex& src) 
{
	dest = make_floatComplex(cuCabsf(src), 0);
}
template<>
__host__ __device__ inline void set_norm<doubleComplex>(doubleComplex& dest, const doubleComplex& src)
{
	dest = make_doubleComplex(cuCabs(src), 0);
}



template<typename T>
__host__ __device__ T inline negone()
{
	return -1;
}
template<>
__host__ __device__ inline floatComplex negone<floatComplex>()
{
	return make_floatComplex(-1,0);
}
template<>
__host__ __device__ inline doubleComplex negone<doubleComplex>()
{
	return make_doubleComplex(-1,0);
}


#include <math.h>
template<typename T>
__host__ __device__ inline void sincosT(T x, T* sin, T* cos)
{
}
template<>
__host__ __device__ inline void sincosT<float>(float x, float* sin, float* cos)
{
	sincosf(x, sin, cos);
}
template<>
__host__ __device__ inline void sincosT<double>(double x, double* sin, double* cos)
{
	sincos(x, sin, cos);
}


