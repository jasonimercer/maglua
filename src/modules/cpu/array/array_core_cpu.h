#include <complex>
#ifndef COMPLEX_TYPES
#define COMPLEX_TYPES
using namespace std;
typedef complex<double> doubleComplex; //cpu version
typedef complex<float>   floatComplex;
#endif


#ifndef ARRAYCORECUDA
#define ARRAYCORECUDA

#include "fourier.h"
#include "luat.h"
#include "array_ops.h"
#include <stdlib.h>

template<typename T>
class ArrayCore
{
public:
	ArrayCore(int x, int y, int z) 
	: 	 fft_plan_1D(0),  fft_plan_2D(0),  fft_plan_3D(0),
		ifft_plan_1D(0), ifft_plan_2D(0), ifft_plan_3D(0),
		nx(x-1), ny(y-1), nz(z-1), _data(0)
	{
		setSize(x,y,z);
	}
	
	~ArrayCore() {setSize(0,0,0);}
	
	void setSize(int x, int y, int z)
	{
		if(x != nx || y != ny || z != nz)
		{
			if( fft_plan_1D) free_FFT_PLAN( fft_plan_1D);
			if( fft_plan_2D) free_FFT_PLAN( fft_plan_2D);
			if( fft_plan_3D) free_FFT_PLAN( fft_plan_3D);
			if(ifft_plan_1D) free_FFT_PLAN(ifft_plan_1D);
			if(ifft_plan_2D) free_FFT_PLAN(ifft_plan_2D);
			if(ifft_plan_3D) free_FFT_PLAN(ifft_plan_3D);

			if(_data) free(_data);
			_data = 0;
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
				_data = (T*)malloc(sizeof(T) * nxyz);
				for(int i=0; i<nxyz; i++)
					_data[i] = luaT<T>::zero();
			}
		}
	}
	
private:
	bool internal_fft(ArrayCore<T>* dest, T* ws, int dims, int direction, FFT_PLAN** plan)
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
	bool fft1DTo(ArrayCore<T>* dest, T* ws){
		return internal_fft(dest, ws, 1, FFT_FORWARD, &fft_plan_1D);
	}
	bool fft2DTo(ArrayCore<T>* dest, T* ws){
		return internal_fft(dest, ws, 2, FFT_FORWARD, &fft_plan_2D);
	}
	bool fft3DTo(ArrayCore<T>* dest, T* ws){
		return internal_fft(dest, ws, 3, FFT_FORWARD, &fft_plan_3D);
	}
	bool ifft1DTo(ArrayCore<T>* dest, T* ws){
		return internal_fft(dest, ws, 1, FFT_BACKWARD, &ifft_plan_1D);
	}
	bool ifft2DTo(ArrayCore<T>* dest, T* ws){
		return internal_fft(dest, ws, 2, FFT_BACKWARD, &ifft_plan_2D);
	}
	bool ifft3DTo(ArrayCore<T>* dest, T* ws){
		return internal_fft(dest, ws, 3, FFT_BACKWARD, &ifft_plan_3D);
	}

	void encode(buffer* b)
	{
		sync_hd();
		encodeInteger(nx, b);
		encodeInteger(ny, b);
		encodeInteger(nz, b);
		for(int i=0; i<nxyz; i++)
			luaT<T>::encode(_data[i], b);
	}
	int decode(buffer* b)
	{
		int x = decodeInteger(b);
		int y = decodeInteger(b);
		int z = decodeInteger(b);
		
		setSize(x,y,z);
		
		for(int i=0; i<nxyz; i++)
			_data[i] = luaT<T>::decode(b);
		return 0;
	}

	
	bool sameSize(const ArrayCore<T>* other) const
	{
		if(!other) return false;
		if(other-> nx != nx) return false;
		if(other-> ny != ny) return false;
		if(other-> nz != nz) return false;
		return true;
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

	int nx, ny, nz, nxyz;
	
	T* data() {return _data;}
	T* _data;
	
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
			for(int i=0; i<3; i++)
			{
				if(lua_isnumber(L, base_idx+i))
				{
					if(lua_isnumber(L, base_idx+i))
					{
						c[i] = lua_tointeger(L, base_idx+i)-1;
						offset++;
					}
				}
			}
		}
		if(!member(c[0], c[1], c[2]))
			return luaL_error(L, "invalid site");
		c[0] = xyz2idx(c[0], c[1], c[2]);
		_data[c[0]] = luaT<T>::to(L, base_idx + offset);
		return 0;
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
		return luaT<T>::push(L, _data[c[0]]);
	}
	
	void setAll(const T& v) {sync_hd(); arraySetAll(_data, nxyz, v);}
	void scaleAll(const T& v) {sync_hd(); arrayScaleAll(_data, nxyz, v);}
	
	static bool doublePrep(ArrayCore<T>* dest, const ArrayCore<T>* src)
	{
		if(!sameSize(dest, src)) return false;
		dest->sync_hd();
		 src->sync_hd();
		return true;
	}
	static bool triplePrep(ArrayCore<T>* dest, const ArrayCore<T>* src1, const ArrayCore<T>* src2)
	{
		if(!doublePrep(dest, src1)) return false;
		if(!doublePrep(dest, src2)) return false;
		return true;
	}
	
	static bool pairwiseMult(ArrayCore<T>* dest, const ArrayCore<T>* src1, const ArrayCore<T>* src2)
	{
		if(!ArrayCore<T>::triplePrep(dest, src1, src2)) return false;
		arrayMultAll(dest->_data, src1->_data, src2->_data, dest->nxyz);
		return true;
	}

	static bool pairwiseDiff(ArrayCore<T>* dest, const ArrayCore<T>* src1, const ArrayCore<T>* src2)
	{
		if(!ArrayCore<T>::triplePrep(dest, src1, src2)) return false;
		arrayDiffAll(dest->_data, src1->_data, src2->_data, dest->nxyz);
		return true;
	}

	static bool norm(ArrayCore<T>* dest, const ArrayCore<T>* src)
	{
		if(!doublePrep(dest, src)) return false;
		arrayNormAll(dest->_data, src->_data, dest->nxyz);
		return true;
	}

	T sum()
	{
		sync_hd(); 
		T v;
		arraySumAll(_data, nxyz, v);
		return v;
	}

	
};

#endif
