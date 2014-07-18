#include "luat.h"
#include <math.h>
#include "array_ops.h"
#include <stdio.h>


template<typename T>
double norm(const T& value) { return 0; }

template <> double norm(const int& value) {return abs(value);}
template <> double norm(const double& value) {return fabs(value);}
template <> double norm(const float& value) {return fabs(value);}
template <> double norm(const doubleComplex& value) {return std::norm(value);}
template <> double norm(const floatComplex& value) {return std::norm(value);}


template<typename T>
void setAll_(T* dest, const int n, const T& v)
{
// #pragma omp for schedule(static)
	for(int i=0; i<n; i++)
		dest[i] = v;
}

void arraySetAll(double* a, const double& v, const int n)
{
	setAll_<double>(a, n, v);
}
void arraySetAll(float* a, const float& v, const int n)
{
	setAll_<float>(a, n, v);
}
void arraySetAll(int* a,  const int& v, const int n)
{
	setAll_<int>(a, n, v);
}
void arraySetAll(doubleComplex* a, const doubleComplex& v, const int n)
{
	setAll_<doubleComplex>(a, n, v);
}
void arraySetAll(floatComplex* a, const floatComplex& v, const int n)
{
	setAll_<floatComplex>(a, n, v);
}




template<typename T>
void arrayChop_(T* dest, const int n, const T& v)
{
	double tol = norm<T>(v);
	for(int i=0; i<n; i++)
	{
		if(norm(dest[i]) < tol)
			dest[i] = luaT<T>::zero();
	}
}

void arrayChop(double* a, const double& v, const int n)
{
	arrayChop_<double>(a, n, v);
}
void arrayChop(float* a, const float& v, const int n)
{
	arrayChop_<float>(a, n, v);
}
void arrayChop(int* a,  const int& v, const int n)
{
	arrayChop_<int>(a, n, v);
}
void arrayChop(doubleComplex* a, const doubleComplex& v, const int n)
{
	arrayChop_<doubleComplex>(a, n, v);
}
void arrayChop(floatComplex* a, const floatComplex& v, const int n)
{
	arrayChop_<floatComplex>(a, n, v);
}




template<typename T>
void scaleAll_o_(T* dest, const int offset, const int n, const T& v)
{
//#pragma omp for schedule(static)
	for(int i=0; i<n; i++)
		dest[i+offset] *= v;
}
void arrayScaleAll(double* a, const double& v, const int n)
{
	scaleAll_o_<double>(a, 0, n, v);
}
void arrayScaleAll(float* a, const float& v, const int n)
{
	scaleAll_o_<float>(a, 0, n, v);
}
void arrayScaleAll(int* a, const int& v, const int n)
{
	scaleAll_o_<int>(a, 0, n, v);
}
void arrayScaleAll(doubleComplex* a, const doubleComplex& v, const int n)
{
	scaleAll_o_<doubleComplex>(a, 0, n, v);
}
void arrayScaleAll(floatComplex* a, const floatComplex& v, const int n)
{
	scaleAll_o_<floatComplex>(a, 0, n, v);
}

void arrayScaleAll_o(double* a, const int offset, const double& v, const int n)
{
	scaleAll_o_<double>(a, offset, n, v);
}
void arrayScaleAll_o(float* a, const int offset, const float& v, const int n)
{
	scaleAll_o_<float>(a, offset, n, v);
}
void arrayScaleAll_o(int* a, const int offset, const int& v, const int n)
{
	scaleAll_o_<int>(a, offset, n, v);
}
void arrayScaleAll_o(doubleComplex* a, const int offset, const doubleComplex& v, const int n)
{
	scaleAll_o_<doubleComplex>(a, offset, n, v);
}
void arrayScaleAll_o(floatComplex* a, const int offset, const floatComplex& v, const int n)
{
	scaleAll_o_<floatComplex>(a, offset, n, v);
}





// dest[i] = (mult1 * src1[i] + add1) * (mult2 * src2[i] + add2)
template<typename T>
void arrayDot_(T* dest, T* s1, T* s2, const int n)
{
//#pragma omp for schedule(static)
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
//#pragma omp for schedule(static)
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
//#pragma omp for schedule(static)
	for(int i=0; i<n; i++)
		d[i] = fabs(s1[i]);
}
void arrayNormAll(float* d, float* s1, const int n)
{
//#pragma omp for schedule(static)
	for(int i=0; i<n; i++)
		d[i] = fabsf(s1[i]);
}
void arrayNormAll(int* d, int* s1, const int n)
{
//#pragma omp for schedule(static)
	for(int i=0; i<n; i++)
	{
		if(d[i] < 0)
			d[i] = -s1[i]; //dealing with problems with intel's compiler
		else
			d[i] = s1[i];
	}
}
void arrayNormAll(doubleComplex* d, doubleComplex* s1, const int n)
{
//#pragma omp for schedule(static)
	for(int i=0; i<n; i++)
		d[i] = doubleComplex(abs(s1[i]), 0);
}
void arrayNormAll(floatComplex* d, floatComplex* s1, const int n)
{
//#pragma omp for schedule(static)
	for(int i=0; i<n; i++)
		d[i] = floatComplex(abs(s1[i]), 0);
}





void arrayPowAll(double* d, double* s1, const double power, const int n)
{
//#pragma omp for schedule(static)
    for(int i=0; i<n; i++)
	d[i] = pow(s1[i], power);
}
void arrayPowAll(float* d, float* s1, const double power, const int n)
{
//#pragma omp for schedule(static)
    for(int i=0; i<n; i++)
	d[i] = pow(s1[i], power);
}
void arrayPowAll(int* d, int* s1, const double power, const int n)
{
//#pragma omp for schedule(static)
    for(int i=0; i<n; i++)
	d[i] = pow(s1[i], power);
}
void arrayPowAll(doubleComplex* d, doubleComplex* s1, const double power, const int n)
{
//#pragma omp for schedule(static)
    for(int i=0; i<n; i++)
	d[i] = pow(s1[i], power);
}
void arrayPowAll(floatComplex* d, floatComplex* s1, const double power, const int n)
{
//#pragma omp for schedule(static)
    for(int i=0; i<n; i++)
	d[i] = pow(s1[i], power);
}






template<typename A, typename B>
void arrayGetRealPart_(A* dest, const B* src, const int n)
{
//#pragma omp for schedule(static)
	for(int i=0; i<n; i++)
		dest[i] = src[i].real();
}
template<typename A, typename B>
void arrayGetImagPart_(A* dest, const B* src, const int n)
{
//#pragma omp for schedule(static)
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
//#pragma omp for schedule(static)
	for(int i=0; i<n; i++)
		dest[i] = A(src[i], dest[i].imag());
}
template<typename A, typename B>
void arraySetImagPart_(A* dest, const B* src, const int n)
{
//#pragma omp for schedule(static)
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
T arrayExtreme_(const T* v, int e, const int n, int& idx)
{
	int _idx = 0;
	T res = v[0];
	if(e == -1)
	{
		for(int i=1; i<n; i++)
			if(luaT<T>::lt(v[i], res))
			{
				res = v[i];
				_idx = i;
			}
	}
	if(e ==  1)
	{
		for(int i=1; i<n; i++)
			if(luaT<T>::lt(res, v[i]))
			{
				res = v[i];
				_idx = i;
			}
	}
	idx = _idx;
	return res;
}

void reduceExtreme(const double* d_a, const int min_max, const int n, double& v, int& idx)
{
	v = arrayExtreme_<double>(d_a, min_max, n, idx);
}
void reduceExtreme(const float* d_a, const int min_max, const int n, float& v, int& idx)
{
	v = arrayExtreme_<float>(d_a, min_max, n, idx);
}
void reduceExtreme(const int* d_a, const int min_max, const int n, int& v, int& idx)
{
	v = arrayExtreme_<int>(d_a, min_max, n, idx);
}
void reduceExtreme(const doubleComplex* d_a, const int min_max, const int n, doubleComplex& v, int& idx)
{
	v = arrayExtreme_<doubleComplex>(d_a, min_max, n, idx);
}
void reduceExtreme(const floatComplex* d_a, const int min_max, const int n, floatComplex& v, int& idx)
{
	v = arrayExtreme_<floatComplex>(d_a, min_max, n, idx);
}



template<typename T>
T arrayPowerSumAll_cplx_(const T* v, const double p, const int n)
{
	T res = 0;
	for(int i=0; i<n; i++)
		res += pow(v[i], p);
	
	return res;
}
template<typename T>
T arrayPowerSumAll_(const T* v, const double p, const int n)
{
	T res = 0;
// #pragma omp parallel for default(shared)	schedule(static) reduction(+:res)  
  for(int i=0; i<n; i++)
	  res += pow((double)v[i], (double)p);
	
	return res;
}



void reducePowerSumAll(const double* a, const double p, const int n, double& v)
{
	v = arrayPowerSumAll_<double>(a, p, n);
}
void reducePowerSumAll(const float* a, const double p, const int n, float& v)
{
	v = arrayPowerSumAll_<float>(a, p, n);
}
void reducePowerSumAll(const int* a, const double p, const int n, int& v)
{
	v = arrayPowerSumAll_<int>(a, p, n);
}
void reducePowerSumAll(const doubleComplex* a, const double p, const int n, doubleComplex& v)
{
	v = arrayPowerSumAll_cplx_<doubleComplex>(a, n, p);
}
void reducePowerSumAll(const floatComplex* a, const double p, const int n, floatComplex& v)
{
	v = arrayPowerSumAll_cplx_<floatComplex>(a, n, p);
}






void reduceDiffSumAll(const double* d_a, const double* d_b, const int n, double& _v)
{
// 	v = 0;
	double v = 0;
//#pragma omp parallel for default(shared) schedule(static) reduction(+:v)  
	for(int i=0; i<n; i++)
	{
		const double q = d_a[i] - d_b[i];
		if(q < 0)
			v += -q;
		else
			v += q;	
	}
	_v = v;
}
void reduceDiffSumAll(const float* d_a, const float* d_b, const int n, float& _v)
{
	float v = 0;
//#pragma omp parallel for default(shared)	schedule(static) reduction(+:v)  
	for(int i=0; i<n; i++)
		v+= fabsf((d_a[i] - d_b[i]));
	_v = v;
}
void reduceDiffSumAll(const int* d_a, const int* d_b, const int n, int& _v)
{
	int v = 0;
//#pragma omp parallel for default(shared)	schedule(static) reduction(+:v)  
	for(int i=0; i<n; i++)
	{
		const int q = d_a[i] - d_b[i];
		if(q < 0)
			v -= q;
		else
			v += q;
	}
	_v = v;
}
void reduceDiffSumAll(const doubleComplex* d_a, const doubleComplex* d_b, const int n, doubleComplex& _v)
{
	doubleComplex v = doubleComplex(0,0);
// #pragma omp parallel for default(shared)	schedule(static) reduction(+:v)  
	for(int i=0; i<n; i++)
		v+= abs((d_a[i] - d_b[i]));
	_v = v;
}
void reduceDiffSumAll(const floatComplex* d_a, const floatComplex* d_b, const int n, floatComplex& _v)
{
	floatComplex v = floatComplex(0,0);
// #pragma omp parallel for default(shared)	schedule(static) reduction(+:v)  
	for(int i=0; i<n; i++)
		v+= abs((d_a[i] - d_b[i]));
	_v = v;
}


void reduceMultSumAll(const double* d_a, const double* d_b, const int n, double& _v)
{
	double v = 0;
//#pragma omp parallel for default(shared)	schedule(static) reduction(+:v)  
	for(int i=0; i<n; i++)
		v+= d_a[i] * d_b[i];
	_v = v;
}
void reduceMultSumAll(const float* d_a, const float* d_b, const int n, float& _v)
{
	float v = 0;
//#pragma omp parallel for default(shared)	schedule(static) reduction(+:v)  
	for(int i=0; i<n; i++)
		v+= d_a[i] * d_b[i];
	_v = v;
}
void reduceMultSumAll(const int* d_a, const int* d_b, const int n, int& _v)
{
	int v = 0;
//#pragma omp parallel for default(shared)	schedule(static) reduction(+:v)  
	for(int i=0; i<n; i++)
		v+= d_a[i] * d_b[i];
	_v = v;
}
void reduceMultSumAll(const doubleComplex* d_a, const doubleComplex* d_b, const int n, doubleComplex& _v)
{
	doubleComplex v = doubleComplex(0,0);
// #pragma omp parallel for default(shared)	schedule(static) reduction(+:v)  
	for(int i=0; i<n; i++)
		v+= d_a[i] * d_b[i];
	_v = v;
}
void reduceMultSumAll(const floatComplex* d_a, const floatComplex* d_b, const int n, floatComplex& _v)
{
	floatComplex v = floatComplex(0,0);
	v = 0;
// #pragma omp parallel for default(shared)	schedule(static) reduction(+:v)  
	for(int i=0; i<n; i++)
		v+= d_a[i] * d_b[i];
	_v = v;
}







void arrayAddAll(double* d_a, const double& v, const int n)
{
//#pragma omp parallel for default(shared)	schedule(static)
	for(int i=0; i<n; i++)
		d_a[i] += v;
}
void arrayAddAll(float* d_a, const float& v, const int n)
{
//#pragma omp parallel for default(shared)	schedule(static)
	for(int i=0; i<n; i++)
		d_a[i] += v;
}
void arrayAddAll(int* d_a, const int& v, const int n)
{
//#pragma omp parallel for default(shared)	schedule(static)
	for(int i=0; i<n; i++)
		d_a[i] += v;
}
void arrayAddAll(doubleComplex* d_a, const doubleComplex& v, const int n)
{
//#pragma omp parallel for default(shared)	schedule(static)
	for(int i=0; i<n; i++)
		d_a[i] += v;
}
void arrayAddAll(floatComplex* d_a, const floatComplex& v, const int n)
{
//#pragma omp parallel for default(shared)	schedule(static)
	for(int i=0; i<n; i++)
		d_a[i] += v;
}



void arrayScaleAdd(double* dest, double s1, const double* src1, double s2, const double* src2, const int n)
{
	for(int i=0; i<n; i++)
		dest[i] = s1*src1[i] + s2*src2[i];
}

void arrayScaleAdd(float* dest, float s1, const float* src1, float s2, const float* src2, const int n)
{
	for(int i=0; i<n; i++)
		dest[i] = s1*src1[i] + s2*src2[i];
}
void arrayScaleAdd(int* dest, int s1, const int* src1, int s2, const int* src2, const int n)
{
	for(int i=0; i<n; i++)
		dest[i] = s1*src1[i] + s2*src2[i];
}
void arrayScaleAdd(doubleComplex* dest, doubleComplex s1, const doubleComplex* src1, doubleComplex s2, const doubleComplex* src2, const int n)
{
	for(int i=0; i<n; i++)
		dest[i] = s1*src1[i] + s2*src2[i];
}

void arrayScaleAdd(floatComplex* dest, floatComplex s1, const floatComplex* src1, floatComplex s2, const floatComplex* src2, const int n)
{
	for(int i=0; i<n; i++)
		dest[i] = s1*src1[i] + s2*src2[i];
}





void arraySumAll(double* d_dest, const double* d_src1, const double* d_src2, const int n)
{
	arrayScaleAdd(d_dest, 1.0, d_src1, 1.0, d_src2, n);
}

void arraySumAll( float* d_dest,  const float* d_src1,  const float* d_src2, const int n)
{
	arrayScaleAdd(d_dest, 1.0, d_src1, 1.0, d_src2, n);
}
void arraySumAll(   int* d_dest,    const int* d_src1,    const int* d_src2, const int n)
{
	arrayScaleAdd(d_dest, 1.0, d_src1, 1.0, d_src2, n);
}
void arraySumAll(doubleComplex* d_dest, const doubleComplex* d_src1, const doubleComplex* d_src2, const int n)
{
	arrayScaleAdd(d_dest, 1.0, d_src1, 1.0, d_src2, n);
}
void arraySumAll( floatComplex* d_dest,  const floatComplex* d_src1,  const floatComplex* d_src2, const int n)
{
	arrayScaleAdd(d_dest, 1.0, d_src1, 1.0, d_src2, n);
}








template<typename T>
void arrayLayerMult_(T* d, int dl, const T* s1, int s1l, const T* s2, int s2l, T mult, const int set, const int nxy)
{
	if(set)
	{
		for(int i=0; i<nxy; i++)
		{
			d[i + dl*nxy]  = s1[i+s1l*nxy] * s2[i+s2l*nxy] * mult;
		}
	}
	else //add
	{
		for(int i=0; i<nxy; i++)
		{
			d[i + dl*nxy] += s1[i+s1l*nxy] * s2[i+s2l*nxy] * mult;
		}
	}
}


void arrayLayerMult(double* dest, int dest_layer, const double* src1, int src1_layer, const double* src2, int src2_layer, double mult, int set, const int nxy)
{
	arrayLayerMult_<double>(dest, dest_layer, src1, src1_layer, src2, src2_layer, mult, set, nxy);
}
void arrayLayerMult(float* dest, int dest_layer, const float* src1, int src1_layer, const float* src2, int src2_layer, float mult, int set, const int nxy)
{
	arrayLayerMult_<float>(dest, dest_layer, src1, src1_layer, src2, src2_layer, mult, set, nxy);
}
void arrayLayerMult(int* dest, int dest_layer, const int* src1, int src1_layer, const int* src2, int src2_layer, int mult, int set, const int nxy)
{
	arrayLayerMult_<int>(dest, dest_layer, src1, src1_layer, src2, src2_layer, mult, set, nxy);
}
void arrayLayerMult(doubleComplex* dest, int dest_layer, const doubleComplex* src1, int src1_layer, const doubleComplex* src2, int src2_layer, doubleComplex mult, int set, const int nxy)
{
	arrayLayerMult_<doubleComplex>(dest, dest_layer, src1, src1_layer, src2, src2_layer, mult, set, nxy);
}
void arrayLayerMult(floatComplex* dest, int dest_layer, const floatComplex* src1, int src1_layer, const floatComplex* src2, int src2_layer, floatComplex mult, int set, const int nxy)
{
	arrayLayerMult_<floatComplex>(dest, dest_layer, src1, src1_layer, src2, src2_layer, mult, set, nxy);
}





template<typename T>
void arrayScaleMultAdd_(T* dest, const int od, T scale, const T* src1, const int o1, const T* src2, const int o2, const T* src3, const int o3, const int nxy)
{
	for(int i=0; i<nxy; i++)
	{
		dest[i+od] = scale * src1[i+o1] * src2[i+o2] + src3[i+o3];
	}
}

ARRAY_API void arrayScaleMultAdd_o(double* dest, const int od, double scale, const double* src1, const int o1, const double* src2, const int o2, const double* src3, const int o3, const int nxy)
{
	arrayScaleMultAdd_<double>(dest, od, scale, src1, o1, src2, o2, src3, o3, nxy);
}

ARRAY_API void arrayScaleMultAdd_o(float* dest, const int od, float scale, const float* src1, const int o1, const float* src2, const int o2, const float* src3, const int o3, const int nxy)
{
	arrayScaleMultAdd_<float>(dest, od, scale, src1, o1, src2, o2, src3, o3, nxy);
}

ARRAY_API void arrayScaleMultAdd_o(int* dest, const int od, int scale, const int* src1, const int o1, const int* src2, const int o2, const int* src3, const int o3, const int nxy)
{
	arrayScaleMultAdd_<int>(dest, od, scale, src1, o1, src2, o2, src3, o3, nxy);
}

ARRAY_API void arrayScaleMultAdd_o(doubleComplex* dest, const int od, doubleComplex scale, const doubleComplex* src1, const int o1, const doubleComplex* src2, const int o2, const doubleComplex* src3, const int o3, const int nxy)
{
	arrayScaleMultAdd_<doubleComplex>(dest, od, scale, src1, o1, src2, o2, src3, o3, nxy);
}

ARRAY_API void arrayScaleMultAdd_o(floatComplex* dest, const int od, floatComplex scale, const floatComplex* src1, const int o1, const floatComplex* src2, const int o2, const floatComplex* src3, const int o3, const int nxy)
{
	arrayScaleMultAdd_<floatComplex>(dest, od, scale, src1, o1, src2, o2, src3, o3, nxy);
}







template<typename T>
void arrayCopyRegionFromTo_(T* src, int sx, int sy, int sz, int* s1, int* s2, T* dest, int dx, int dy, int dz, int* d1, int* d2)
{
	int dim1 = s2[0] - s1[0];
	int dim2 = s2[1] - s1[1];
	int dim3 = s2[2] - s1[2];
	
	for(int k=0; k<=dim3; k++)
		for(int j=0; j<=dim2; j++)
			for(int i=0; i<=dim1; i++)
			{
				dest[(i + d1[0]) + (j + d1[1]) * dx + (k + d1[2]) * dx*dy] = src[(i + s1[0]) + (j + s1[1]) * sx + (k + s1[2]) * sx*sy];
			}
}





ARRAY_API void arrayCopyRegionFromTo(double* src, int sx, int sy, int sz, int* s1, int* s2, double* dest, int dx, int dy, int dz, int* d1, int* d2)
{
	arrayCopyRegionFromTo_<double>(src, sx,sy,sz, s1,s2,dest,dx,dy,dz, d1, d2);
}
ARRAY_API void arrayCopyRegionFromTo( float* src, int sx, int sy, int sz, int* s1, int* s2,  float* dest, int dx, int dy, int dz, int* d1, int* d2)
{
	arrayCopyRegionFromTo_<float>(src, sx,sy,sz, s1,s2,dest,dx,dy,dz, d1, d2);
}
ARRAY_API void arrayCopyRegionFromTo(   int* src, int sx, int sy, int sz, int* s1, int* s2,    int* dest, int dx, int dy, int dz, int* d1, int* d2)
{
	arrayCopyRegionFromTo_<int>(src, sx,sy,sz, s1,s2,dest,dx,dy,dz, d1, d2);
}
ARRAY_API void arrayCopyRegionFromTo(doubleComplex* src, int sx, int sy, int sz, int* s1, int* s2, doubleComplex* dest, int dx, int dy, int dz, int* d1, int* d2)
{
	arrayCopyRegionFromTo_<doubleComplex>(src, sx,sy,sz, s1,s2,dest,dx,dy,dz, d1, d2);
}
ARRAY_API void arrayCopyRegionFromTo( floatComplex* src, int sx, int sy, int sz, int* s1, int* s2,  floatComplex* dest, int dx, int dy, int dz, int* d1, int* d2)
{
	arrayCopyRegionFromTo_<floatComplex>(src, sx,sy,sz, s1,s2,dest,dx,dy,dz, d1, d2);
}

