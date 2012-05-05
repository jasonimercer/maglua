#include <cuComplex.h>
typedef cuDoubleComplex doubleComplex; //cuda version
typedef cuFloatComplex floatComplex;

#ifndef ARRAYCORECUDA
#define ARRAYCORECUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include "luat.h"
#include "array_ops.hpp"
#include "memory.hpp"
#include "fourier_kernel.hpp"
#include <stdlib.h>

template<typename T>
class ArrayCore
{
public:
	ArrayCore(int x, int y, int z) 
	: 	 fft_plan_1D(0),  fft_plan_2D(0),  fft_plan_3D(0),
		ifft_plan_1D(0), ifft_plan_2D(0), ifft_plan_3D(0),
		nx(x-1), ny(y-1), nz(z-1), h_data(0), d_data(0)
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

			if(d_data) free_device(d_data);
			if(h_data) free_host(h_data);
			d_data = 0;
			h_data = 0;
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
// 				malloc_device(&d_data, sizeof(T) * nxyz); //going to malloc device lazily
				malloc_host(&h_data, sizeof(T) * nxyz);
				for(int i=0; i<nxyz; i++)
					h_data[i] = luaT<T>::zero();
				new_host = true;
				new_device = false;
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
		sync_hd(); dest->sync_hd();
		execute_FFT_PLAN(*plan, dest->d_data, d_data, ws);
		dest->new_device = true;
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
			luaT<T>::encode(h_data[i], b);
	}
	int decode(buffer* b)
	{
		int x = decodeInteger(b);
		int y = decodeInteger(b);
		int z = decodeInteger(b);
		
		setSize(x,y,z);
		
		for(int i=0; i<nxyz; i++)
			h_data[i] = luaT<T>::decode(b);

		new_host = true;
		new_device = false;
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
	
	T* data() {if(d_data); sync_dh(); return h_data;}
	T* h_data;
	T* d_data;
	
	//sync host to device (if needed)
	void sync_hd()
	{
		if(!new_host) return;
		if(!d_data)
			malloc_device(&d_data, sizeof(T) * nxyz);
		memcpy_h2d(d_data, h_data, sizeof(T)*nxyz);
		new_host = false;
	}
	
	 //sync device to host (if needed)
	void sync_dh()
	{
		if(!new_device) return;
		if(!d_data) return;
		memcpy_d2h(h_data, d_data, sizeof(T)*nxyz);
		new_device = false;
	}
	
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
		h_data[c[0]] = luaT<T>::to(L, base_idx + offset);
		new_host = true;
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
		return luaT<T>::push(L, h_data[c[0]]);
	}
	
	void setAll(const T& v) {sync_hd(); arraySetAll(d_data, nxyz, v); new_device=true;}
	void scaleAll(const T& v) {sync_hd(); arrayScaleAll(d_data, nxyz, v); new_device=true;}
	
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
		arrayMultAll(dest->d_data, src1->d_data, src2->d_data, dest->nxyz);
		dest->new_device=true;
		return true;
	}

	static bool pairwiseDiff(ArrayCore<T>* dest, const ArrayCore<T>* src1, const ArrayCore<T>* src2)
	{
		if(!ArrayCore<T>::triplePrep(dest, src1, src2)) return false;
		arrayDiffAll(dest->d_data, src1->d_data, src2->d_data, dest->nxyz);
		dest->new_device=true;
		return true;
	}

	static bool norm(ArrayCore<T>* dest, const ArrayCore<T>* src)
	{
		if(!doublePrep(dest, src)) return false;
		arrayNormAll(dest->d_data, src->d_data, dest->nxyz);
		dest->new_device=true;
		return true;
	}

	T sum()
	{
		sync_hd(); 
		T v;
		arraySumAll(d_data, nxyz, v);
		return v;
	}



	// flags indicating if there is new data on host/device
	bool new_device;
	bool new_host;
	

	
};

#endif
