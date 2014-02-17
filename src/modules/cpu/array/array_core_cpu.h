#ifndef ARRAYCORECPU
#define ARRAYCORECPU

#include "fourier.h"
#include "luat.h"
#include "array_ops.h"
#include <stdlib.h>
#include "luabaseobject.h"
#include <fftw3.h>
#include "memory.h"

template<typename T>
inline const char* array_lua_name() {return "Array.Unnamed";}

template<>inline const char* array_lua_name<int>() {return "Array.Integer";}
template<>inline const char* array_lua_name<float>() {return "Array.Float";}
template<>inline const char* array_lua_name<double>() {return "Array.Double";}
template<>inline const char* array_lua_name<floatComplex>() {return "Array.FloatComplex";}
template<>inline const char* array_lua_name<doubleComplex>() {return "Array.DoubleComplex";}


template<typename T>
class ARRAY_API Array;

template<typename T>
inline int array_help_specialization(lua_State* L){return 0;}

template<typename T>
inline const luaL_Reg* array_luamethods_specialization(){return 0;}

template<typename T>
inline int array_luainit_specialization(Array<T>* that, lua_State* L){ return 0;}



// these are specialization functions that will bridge from this header
// to implementation on the source side.
#define array_tspec(type) \
int array_help_specialization_ ## type (lua_State* L); \
template<> inline int array_help_specialization< type >(lua_State* L){ \
return array_help_specialization_## type (L); } \
\
const luaL_Reg* array_luamethods_specialization_ ## type (); \
template<> inline const luaL_Reg* array_luamethods_specialization<type>(){ \
return array_luamethods_specialization_ ## type ();} \
\
int array_luainit_specialization_ ## type(Array< type >* that, lua_State* L); \
template<> inline int array_luainit_specialization<type>(Array< type >* that, lua_State* L){ \
return array_luainit_specialization_ ## type (that, L);}		\


array_tspec(int);
array_tspec(float);
array_tspec(double);
array_tspec(floatComplex);
array_tspec(doubleComplex);

template<typename T>
class ARRAY_API Array : public LuaBaseObject	
{
public:
	LINEAGE1(array_lua_name<T>())
	static const luaL_Reg* luaMethods()
	{
		return array_luamethods_specialization<T>();
	} 
	virtual int luaInit(lua_State* L)
	{
	  return array_luainit_specialization<T>(this, L);
	}
	static int help(lua_State* L)
	{
		return array_help_specialization<T>(L);
	}

	
	Array(int x=4, int y=4, int z=1, T* memory=0) 
	: 	 fft_plan_1D(0),  fft_plan_2D(0),  fft_plan_3D(0),
		ifft_plan_1D(0), ifft_plan_2D(0), ifft_plan_3D(0),
		nx(x-1), ny(y-1), nz(z-1), _data(0), LuaBaseObject(hash32((array_lua_name<T>())))
	{
		i_own_my_memory = (memory == 0);
		setSize(x,y,z,memory);
	}
	
	~Array() {setSize(0,0,0);}
	
	void setSize(int x, int y, int z, T* use_this_memory=0)
	{
		if(x <= 0) x = 1;
		if(y <= 0) y = 1;
		if(z <= 0) z = 1;
		if(x != nx || y != ny || z != nz)
		{
			if( fft_plan_1D) free_FFT_PLAN( fft_plan_1D);
			if( fft_plan_2D) free_FFT_PLAN( fft_plan_2D);
			if( fft_plan_3D) free_FFT_PLAN( fft_plan_3D);
			if(ifft_plan_1D) free_FFT_PLAN(ifft_plan_1D);
			if(ifft_plan_2D) free_FFT_PLAN(ifft_plan_2D);
			if(ifft_plan_3D) free_FFT_PLAN(ifft_plan_3D);

			//if(_data) free(_data);
			if(i_own_my_memory)
				if(_data)
					fftw_free(_data);
			_data = use_this_memory;
			fft_plan_1D = 0;
			fft_plan_2D = 0;
			fft_plan_3D = 0;
			ifft_plan_1D = 0;
			ifft_plan_2D = 0;
			ifft_plan_3D = 0;
			
			nx = x; ny = y; nz = z;
			nxyz = nx*ny*nz;
			if(nxyz)
			{
				if(!_data)
					_data = (T*)fftw_malloc(sizeof(T) * nxyz);
				zero();
			}
		}
	}
	
private:
	bool internal_fft(Array<T>* dest, T* ws, int dims, int direction, FFT_PLAN** plan)
	{
		if(!sameSize(dest)) return false;
		if(!*plan)
		{
			*plan = make_FFT_PLAN_T<T>(direction, dims, nx, ny, nz);
			if(!*plan)
				return false;
		}
		//sync_hd(); dest->sync_hd();
		execute_FFT_PLAN(*plan, dest->_data, _data, ws);
		return true;
	}
	FFT_PLAN* fft_plan_1D;
	FFT_PLAN* fft_plan_2D;
	FFT_PLAN* fft_plan_3D;
	
	FFT_PLAN* ifft_plan_1D;
	FFT_PLAN* ifft_plan_2D;
	FFT_PLAN* ifft_plan_3D;

public:
	bool fft1DTo(Array<T>* dest, T* ws=0){
		return internal_fft(dest, ws, 1, FFT_FORWARD, &fft_plan_1D);
	}
	bool fft1DTo(Array<T>* dest, Array<T>* ws){
		return fft1DTo(dest,ws->data());
	}
	bool fft2DTo(Array<T>* dest, T* ws=0){
		return internal_fft(dest, ws, 2, FFT_FORWARD, &fft_plan_2D);
	}
	bool fft2DTo(Array<T>* dest, Array<T>* ws){
		return fft2DTo(dest,ws->data());
	}
	bool fft3DTo(Array<T>* dest, T* ws=0){
		return internal_fft(dest, ws, 3, FFT_FORWARD, &fft_plan_3D);
	}
	bool fft3DTo(Array<T>* dest, Array<T>* ws){
		return fft3DTo(dest,ws->data());
	}
	
	bool ifft1DTo(Array<T>* dest, T* ws=0){
		return internal_fft(dest, ws, 1, FFT_BACKWARD, &ifft_plan_1D);
	}
	bool ifft1DTo(Array<T>* dest, Array<T>* ws){
		return ifft1DTo(dest,ws->data());
	}
	bool ifft2DTo(Array<T>* dest, T* ws=0){
		return internal_fft(dest, ws, 2, FFT_BACKWARD, &ifft_plan_2D);
	}
	bool ifft2DTo(Array<T>* dest, Array<T>* ws){
		return ifft2DTo(dest,ws->data());
	}
	bool ifft3DTo(Array<T>* dest, T* ws=0){
		return internal_fft(dest, ws, 3, FFT_BACKWARD, &ifft_plan_3D);
	}
	bool ifft3DTo(Array<T>* dest, Array<T>* ws){
		return ifft3DTo(dest,ws->data());
	}

	bool areAllSameValue(T& v)
	{
		sync_hd();
		v = _data[0];
		for(int i=1; i<nxyz; i++)
		{
			if(v != _data[i])
				return false;
		}
		return true;
	}

	void encode(buffer* b)
	{
		ENCODE_PREAMBLE
		sync_hd();
		char version = 0;
		encodeChar(version, b);
		encodeInteger(nx, b);
		encodeInteger(ny, b);
		encodeInteger(nz, b);
		int flag = 0;
		T v;
		if(areAllSameValue(v))
		{
			flag = 1;
			encodeInteger(flag, b);
			luaT<T>::encode(v, b);
		}
		else
		{
			flag = 0;
			encodeInteger(flag, b);
			encodeBuffer(_data, sizeof(T)*nxyz, b);
			//for(int i=0; i<nxyz; i++)
			//	luaT<T>::encode(_data[i], b);
		}
	}
	int decode(buffer* b)
	{
		char version = decodeChar(b);
		if(version == 0)
		{
			int x    = decodeInteger(b);
			int y    = decodeInteger(b);
			int z    = decodeInteger(b);
			int flag = decodeInteger(b);
			
			setSize(x,y,z);
			
			if(flag == 1)
			{
				setAll( luaT<T>::decode(b) );
			}
			else
			{
				decodeBuffer(_data, sizeof(T)*nxyz, b);
			}
		}
		else
		{
			fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
		}
		return 0;
	}
	
	bool sameSize(const Array<T>* other) const
	{
		if(!other) return false;
		if(other-> nx != nx) return false;
		if(other-> ny != ny) return false;
		if(other-> nz != nz) return false;
		return true;
	}
	
	// s1,s2 are inclusive
	void copyRegionFromTo(int* s1, int* s2, Array<T>* dest, int* d1, int* d2)
	{
		arrayCopyRegionFromTo(data(), nx, ny, nz, s1, s2, dest->data(), dest->nx, dest->ny, dest->nz, d1, d2);
	}

	int xyz2idx(const int x, const int y, const int z) const
	{
		return x + y*nx + z*nx*ny;
	}
	
	bool member(const int x, const int y, const int z) const
	{
		if(x<0 || x>=nx) return false;
		if(y<0 || y>=ny) return false;
		if(z<0 || z>=nz) return false;
		return true;
	}
	
	bool member(const int x, const int y, const int z, int& idx) const
	{
		if(x<0 || x>=nx) return false;
		if(y<0 || y>=ny) return false;
		if(z<0 || z>=nz) return false;
		idx = xyz2idx(x,y,z);
		return true;
	}

	int nx, ny, nz, nxyz;
	
	T* data() {return _data;}
	T* ddata() {return _data;} // to make things work nicely with the gpu code
	const T* constData() const {return _data;}
	T* _data;
	bool i_own_my_memory;

	void sync_hd() {};
	void sync_dh() {};
	
	int lua_set(lua_State* L, int base_idx)
	{
		sync_dh();
		int c[3] = {0,0,0};
		int offset = 0;
		if(lua_istable(L, base_idx))
		{
			for(int i=0; i<3; i++)
			{
				lua_pushinteger(L, i+1);
				lua_gettable(L, base_idx);
				if(lua_isnumber(L, -1))
					c[i] = lua_tointeger(L, -1)-1;
				lua_pop(L, 1);
			}
			offset = 1;
		}
		else
		{
			offset = 0;
			const int e = luaT<T>::elements();
			const int end = lua_gettop(L) - e;
			int v = 0;
			for(int i=base_idx; i<=end && v<3; i++)
			{
				if(lua_isnumber(L, i))
				{
					c[v] = lua_tointeger(L, i)-1;
					offset++;
					v++;
				}
			}
		}
		if(!member(c[0], c[1], c[2]))
			return luaL_error(L, "invalid site");
		c[0] = xyz2idx(c[0], c[1], c[2]);
		data()[c[0]] = luaT<T>::to(L, base_idx + offset);
		return 0;
	}
	
		
	int lua_addat(lua_State* L, int base_idx)
	{
		sync_dh();
		int c[3] = {0,0,0};
		int offset = 0;
		if(lua_istable(L, base_idx))
		{
			for(int i=0; i<3; i++)
			{
				lua_pushinteger(L, i+1);
				lua_gettable(L, base_idx);
				if(lua_isnumber(L, -1))
					c[i] = lua_tointeger(L, -1)-1;
				lua_pop(L, 1);
			}
			offset = 1;
		}
		else
		{
			offset = 0;
			const int e = luaT<T>::elements();
			const int end = lua_gettop(L) - e;
			int v = 0;
			for(int i=base_idx; i<=end && v<3; i++)
			{
				if(lua_isnumber(L, i))
				{
					c[v] = lua_tointeger(L, i)-1;
					offset++;
					v++;
				}
			}
		}
		if(!member(c[0], c[1], c[2]))
			return luaL_error(L, "invalid site");
		c[0] = xyz2idx(c[0], c[1], c[2]);
		data()[c[0]] += luaT<T>::to(L, base_idx + offset);
		return 0;
	}
	
	T get(int x=0, int y=0, int z=0)
	{
		return data()[xyz2idx(x,y,z)];
	}
	
	void set(int x, int y, int z, T v)
	{
		data()[xyz2idx(x,y,z)] = v;
	}
	
	int lua_get(lua_State* L, int base_idx)
	{
		sync_dh();
		int c[3] = {0,0,0};
		if(lua_istable(L, base_idx))
		{
			for(int i=0; i<3; i++)
			{
				lua_pushinteger(L, i+1);
				lua_gettable(L, base_idx);
				if(lua_isnumber(L, -1))
				{
					c[i] = lua_tointeger(L, -1)-1;
				}
				lua_pop(L, 1);
			}
		}
		else
		{
			for(int i=0; i<3; i++)
			{
				if(lua_isnumber(L, base_idx+i))
					c[i] = lua_tointeger(L, base_idx+i)-1;
			}
		}
		if(!member(c[0], c[1], c[2]))
			return luaL_error(L, "invalid site");
		c[0] = xyz2idx(c[0], c[1], c[2]);
		return luaT<T>::push(L, data()[c[0]]);
	}
	
	void setAll(const T& v) {sync_hd(); arraySetAll(data(), v, nxyz);}
	void scaleAll(const T& v) {sync_hd(); arrayScaleAll(data(), v, nxyz);}
	void scaleAll_o(const T& v, const int offset, const int n) {sync_hd(); arrayScaleAll_o(data(), offset, v, n);}
	void addValue(const T& v) {sync_hd(); arrayAddAll(data(), v, nxyz);}
	
	
	static bool doublePrep(Array<T>* dest, const Array<T>* src)
	{
		if(!dest->sameSize(src)) return false;
		return true;
	}
	static bool triplePrep(Array<T>* dest, const Array<T>* src1, const Array<T>* src2)
	{
		if(!doublePrep(dest, src1)) return false;
		if(!doublePrep(dest, src2)) return false;
		return true;
	}
	
	static bool pairwiseMult(Array<T>* dest, Array<T>* src1, Array<T>* src2)
	{
		if(!Array<T>::triplePrep(dest, src1, src2)) return false;
		arrayMultAll(dest->data(), src1->data(), src2->data(), dest->nxyz);
		return true;
	}

	static bool pairwiseDiff(Array<T>* dest, Array<T>* src1, Array<T>* src2)
	{
		if(!Array<T>::triplePrep(dest, src1, src2)) return false;
		arrayDiffAll(dest->data(), src1->data(), src2->data(), dest->nxyz);
		return true;
	}
	
	static bool pairwiseScaleAdd(Array<T>* dest, const T& s1, Array<T>* src1, const T& s2, Array<T>* src2)
	{
		if(!Array<T>::triplePrep(dest, src1, src2)) return false;
		arrayScaleAdd(dest->data(), s1, src1->constData(), s2, src2->constData(), dest->nxyz);
		return true;
	}

	static bool norm(Array<T>* dest, Array<T>* src)
	{
		if(!doublePrep(dest, src)) return false;
		arrayNormAll(dest->_data, src->_data, dest->nxyz);
		return true;
	}
	
	static T dot(Array<T>* a, Array<T>* b)
	{
		if(!doublePrep(a, b)) return luaT<T>::zero();
		T t;
		reduceMultSumAll(a->_data, b->_data, a->nxyz, t);
		return t;
	}

	T max(int& idx)
	{
		T v;
		reduceExtreme(_data, 1, nxyz, v, idx);
		return v;
	}
	T mean()
	{
		T v;
		reducePowerSumAll(_data, 1.0, nxyz, v);
		return v / T(nxyz);
	}
	T min(int& idx)
	{
		T v;
		reduceExtreme(_data,-1, nxyz, v, idx);
		return v;
	}

	T sum(double power=1.0)
	{
		sync_hd(); 
		T v;
		reducePowerSumAll(_data, power, nxyz, v);
		return v;
	}

	T diffSum(Array<T>* other)
	{
		sync_hd(); 
		T v;
		reduceDiffSumAll(data(), other->data(), nxyz, v);
		return v;
	}
	
	void copyFrom(Array<T>* other)
	{
		if(other != this)
			memcpy(data(), other->data(), sizeof(T)*nxyz);
	}
	
	void zero()
	{
		arraySetAll(data(),  luaT<T>::zero(), nxyz);
	}

	Array<T>& operator+=(const Array<T> &rhs)
	{
		arraySumAll(data(), data(), rhs.constData(), nxyz);
		return *this;
	}
	T& operator[](int index){
		return data()[index];
	}
};

#ifdef WIN32
#ifndef DUMMYWINDOWS_ARRAY_INSTANTIATION
#define DUMMYWINDOWS_ARRAY_INSTANTIATION
//forcing instantiation so they get exported
template class Array<doubleComplex>;
template class Array<double>;
template class Array<floatComplex>;
template class Array<float>;
template class Array<int>;
#endif
#endif



// template class Array<doubleComplex>;
// template class Array<double>;
// template class Array<floatComplex>;
// template class Array<float>;
// template class Array<int>;

#if 0
#define inst_funcs(name)		      \
template int name<Array<float> >(lua_State*); \
template int name<Array<double> >(lua_State*); \
template int name<Array<int> >(lua_State*); \
template int name<Array<floatComplex> >(lua_State*); \
template int name<Array<doubleComplex> >(lua_State*);

inst_funcs(luaT_new)
inst_funcs(luaT_gc)
inst_funcs(luaT_help)
inst_funcs(luaT_mt)


template LuaBaseObject* new_luabaseobject<Array<double> >();
template LuaBaseObject* new_luabaseobject<Array<float> >();
template LuaBaseObject* new_luabaseobject<Array<doubleComplex> >();
template LuaBaseObject* new_luabaseobject<Array<floatComplex> >();
template LuaBaseObject* new_luabaseobject<Array<int> >();


template void luaT_push<Array<double> >(lua_State*, LuaBaseObject*);
template void luaT_push<Array<float > >(lua_State*, LuaBaseObject*);
template void luaT_push<Array<doubleComplex> >(lua_State*, LuaBaseObject*);
template void luaT_push<Array<floatComplex> >(lua_State*, LuaBaseObject*);
template void luaT_push<Array<int> >(lua_State*, LuaBaseObject*);
#endif



typedef Array<doubleComplex> dcArray;
typedef Array<floatComplex>  fcArray;
typedef Array<double>         dArray;
typedef Array<float>          fArray;
typedef Array<int>            iArray;


ARRAY_API dcArray* getWSdcArray(int nx, int ny, int nz, long level);
ARRAY_API fcArray* getWSfcArray(int nx, int ny, int nz, long level);
ARRAY_API dArray* getWSdArray(int nx, int ny, int nz, long level);
ARRAY_API fArray* getWSfArray(int nx, int ny, int nz, long level);
ARRAY_API iArray* getWSiArray(int nx, int ny, int nz, long level);

#endif
