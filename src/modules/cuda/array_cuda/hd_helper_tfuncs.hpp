#include <cuComplex.h>
typedef cuDoubleComplex doubleComplex; //cuda version
typedef cuFloatComplex floatComplex;


template<typename T>
__host__ __device__ inline T subtract(const T& lhs, const T& rhs)
{
	return lhs - rhs;
}
template<>
__host__ __device__ inline floatComplex subtract<floatComplex>(const floatComplex& lhs, const floatComplex& rhs)
{
	return cuCsubf(lhs, rhs);
}
template<>
__host__ __device__ inline doubleComplex subtract<doubleComplex>(const  doubleComplex& lhs, const doubleComplex& rhs)
{
	return cuCsub(lhs, rhs);
}



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
	return make_cuFloatComplex(0.0,0.0);
}
template<>
__host__ __device__ inline doubleComplex zero<doubleComplex>()
{
	return make_cuDoubleComplex(0.0,0.0);
}


template<typename T>
__host__ __device__ inline T one()
{
	return 1;
}
template<>
__host__ __device__ inline floatComplex one<floatComplex>()
{
	return make_cuFloatComplex(1.0,0.0);
}
template<>
__host__ __device__ inline doubleComplex one<doubleComplex>()
{
	return make_cuDoubleComplex(1.0,0.0);
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
	dest = make_cuFloatComplex(cuCabsf(src), 0.0);
}
template<>
__host__ __device__ inline void set_norm<doubleComplex>(doubleComplex& dest, const doubleComplex& src)
{
	dest = make_cuDoubleComplex(cuCabs(src), 0.0);
}



template<typename T>
__host__ __device__ T inline negone()
{
	return -1;
}
template<>
__host__ __device__ inline floatComplex negone<floatComplex>()
{
	return make_cuFloatComplex(-1.0,0.0);
}
template<>
__host__ __device__ inline doubleComplex negone<doubleComplex>()
{
	return make_cuDoubleComplex(-1.0,0.0);
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







template<typename T>
__host__ __device__ inline bool less_than(const T& a, const T& b)
{
	return false;
}
template<>
__host__ __device__ inline bool less_than<int>(const int& a, const int& b) 
{
	return a < b;
}
template<>
__host__ __device__ inline bool less_than<float>(const float& a, const float& b) 
{
	return a < b;
}
template<>
__host__ __device__ inline bool less_than<double>(const double& a, const double& b) 
{
	return a < b;
}
template<>
__host__ __device__ inline bool less_than<floatComplex>(const floatComplex& a, const floatComplex& b) 
{
	return a.x*a.x + a.y*a.y < b.x*b.x + b.y*b.y;
}
template<>
__host__ __device__ inline bool less_than<doubleComplex>(const doubleComplex& a, const doubleComplex& b)
{
	return a.x*a.x + a.y*a.y < b.x*b.x + b.y*b.y;
}







template<typename T>
__host__ __device__ inline void divide(T& dest, const T& a, const T& b)
{
	dest = a/b;	
}
template<>
__host__ __device__ inline void divide<floatComplex>(floatComplex& dest, const floatComplex& v1, const floatComplex& v2)
{
	const float a  = v1.x;
	const float b  = v1.y;
	const float c  = v2.x;
	const float d  = v2.y;
	
	const float num_r = a*c+b*d;
	const float num_i = b*c-a*d;
	
	const float denom = c*c+d*d;
	
	dest.x = num_r / denom;
	dest.y = num_i / denom;
}
template<>
__device__ inline void divide<doubleComplex>(doubleComplex& dest, const doubleComplex& v1, const doubleComplex& v2)
{
	const double a  = v1.x;
	const double b  = v1.y;
	const double c  = v2.x;
	const double d  = v2.y;
	
	const double num_r = a*c+b*d;
	const double num_i = b*c-a*d;
	
	const double denom = c*c+d*d;
	
	dest.x = num_r / denom;
	dest.y = num_i / denom;
}




template<typename T>
__host__ __device__ inline void divide_real(T& dest, const T& a, double b)
{
	dest = a/b;
}
template<>
__host__ __device__ inline void divide_real<floatComplex>(floatComplex& dest, const floatComplex& a, double b)
{
	dest.x = a.x / b;
	dest.y = a.y / b;
}
template<>
__device__ inline void divide_real<doubleComplex>(doubleComplex& dest, const doubleComplex& a, double b)
{
	dest.x = a.x / b;
	dest.y = a.y / b;
}






template<typename T>
__host__ __device__ inline bool equal(const T& lhs, const T& rhs)
{
	return lhs == rhs;
}
template<>
__host__ __device__ inline bool equal<floatComplex>(const floatComplex& lhs, const floatComplex& rhs)
{
	return lhs.x == rhs.x && lhs.y == rhs.y;
}
template<>
__device__ inline bool equal<doubleComplex>(const doubleComplex& lhs, const doubleComplex& rhs)
{
	return lhs.x == rhs.x && lhs.y == rhs.y;
}

