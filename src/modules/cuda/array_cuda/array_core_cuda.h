// This templated class has gotten a bit complicated. This class 
// provides the core of the Array class. This implementation allows
// nearly seamless translation between GPU and Main memory via sync 
// calls. Care must be used when making changes directly to [d]data()
// as the "dirty" flags (new_device, new_host) must be set.

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
#include "hd_helper_tfuncs.hpp"
#include <stdlib.h>

#include "luabaseobject.h"

template<typename T>
inline const char* array_lua_name() {return "Array.Unnamed";}

template<>inline const char* array_lua_name<int>() {return "Array.Integer";}
template<>inline const char* array_lua_name<float>() {return "Array.Float";}
template<>inline const char* array_lua_name<double>() {return "Array.Double";}
template<>inline const char* array_lua_name<floatComplex>() {return "Array.FloatComplex";}
template<>inline const char* array_lua_name<doubleComplex>() {return "Array.DoubleComplex";}

#define ARRAYCUDA_API

template<typename T>
class ARRAYCUDA_API Array : public LuaBaseObject	
{
public:
	LINEAGE1(array_lua_name<T>())
	static const luaL_Reg* luaMethods(); 
	virtual int luaInit(lua_State* L); 
	static int help(lua_State* L); 	

	Array(int x=1, int y=1, int z=1, T* device_memory=0, T* host_memory=0) 
	: 	 fft_plan_1D(0),  fft_plan_2D(0),  fft_plan_3D(0),
		ifft_plan_1D(0), ifft_plan_2D(0), ifft_plan_3D(0),
		nx(x-1), ny(y-1), nz(z-1), h_data(0), d_data(0),ws(0),  //offsets on sizes to make sure "old" sizes are different
		LuaBaseObject(hash32((array_lua_name<T>())))
	{
		registerWS();
		i_own_my_device_memory = (device_memory == 0);
		i_own_my_host_memory = (host_memory == 0);
		setSize(x,y,z, device_memory, host_memory);
	}
	
	~Array()
	{
		unregisterWS();
		setSize(0,0,0);
	}
	 
	void setSize(int x, int y, int z, T* use_this_device_memory = 0, T* use_this_host_memory = 0)
	{
		if(x != nx || y != ny || z != nz)
		{
			if( fft_plan_1D) free_FFT_PLAN( fft_plan_1D);
			if( fft_plan_2D) free_FFT_PLAN( fft_plan_2D);
			if( fft_plan_3D) free_FFT_PLAN( fft_plan_3D);
			if(ifft_plan_1D) free_FFT_PLAN(ifft_plan_1D);
			if(ifft_plan_2D) free_FFT_PLAN(ifft_plan_2D);
			if(ifft_plan_3D) free_FFT_PLAN(ifft_plan_3D);

			if(i_own_my_device_memory)
				if(d_data) free_device(d_data);
			if(i_own_my_host_memory)
				if(h_data) free_host(h_data);
			d_data = use_this_device_memory;
			h_data = use_this_host_memory;
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
				//malloc_device(&d_data, sizeof(T) * nxyz); //going to malloc device lazily
				if(!h_data)
					malloc_host(&h_data, sizeof(T) * nxyz);
				for(int i=0; i<nxyz; i++)
					h_data[i] = luaT<T>::zero();
				new_host = true;
				new_device = false;
				
				getWSMemD(&ws, sizeof(T) * nxyz, hash32("Array_WS"));
				//printf("New array %p %p %p\n", d_data, h_data, ws);
			}
		}
	}
	
private:
	bool internal_fft(Array<T>* dest, int dims, int direction, FFT_PLAN** plan)
	{
		if(ws == 0)
		{
			fprintf(stderr, "(%s:%i) CUDA FFT requires workspace\n", __FILE__, __LINE__);
			return false;
		}
		if(!sameSize(dest)) return false;
		if(!*plan)
		{
			*plan = make_FFT_PLAN_T<T>(direction, dims, nx, ny, nz);
			if(!*plan)
				return false;
		}
		//sync_hd(); dest->sync_hd(); //sync in ddata()
		execute_FFT_PLAN(*plan, dest->ddata(), ddata(), ws);
		dest->new_device = true;
		return true;
	}
	FFT_PLAN* fft_plan_1D;
	FFT_PLAN* fft_plan_2D;
	FFT_PLAN* fft_plan_3D;
	
	FFT_PLAN* ifft_plan_1D;
	FFT_PLAN* ifft_plan_2D;
	FFT_PLAN* ifft_plan_3D;
	
	bool i_own_my_device_memory;
	bool i_own_my_host_memory;

	T* h_data;
	T* d_data;
	
	T* ws; //work space for FFT and other operations

public:
	// flags indicating if there is new data on host/device
	bool new_device;
	bool new_host;
	
	int nx, ny, nz, nxyz;
	
	// get the host memory:
	// the sync_dh doesn't sync if there isn't any device memory 
	// allocated (which is the desired action)
	T*  data() {sync_dh(); return h_data;}

	// get the device memory:
	// this call will sync any new host memory before returning
	// device memory pointer. If "false" is passed in then
	// this will return a null pointer when device memory isn't allocated
	T* ddata(bool allocate_if_needed=true)
	{
		if(!d_data && allocate_if_needed)
		{
			malloc_device(&d_data, sizeof(T) * nxyz);
			i_own_my_device_memory = true;
			new_host = true;
			new_device = false;
		}
		sync_hd(); 
		return d_data;
	}
	
	//sync host to device (if needed)
	void sync_hd()
	{
		if(!d_data)
		{
			new_host = false;
			return;
		}
		if(!new_host)
		{
			return;
		}
		memcpy_h2d(d_data, h_data, sizeof(T)*nxyz);
		new_host = false;
		new_device = false;
	}
	
	// sync device to host (if needed)
	// if there is no device memory then nothing is done
	void sync_dh()
	{
		if(!new_device)
		{
			return;
		}
		if(!d_data)
		{
			return;
		}
		memcpy_d2h(h_data, d_data, sizeof(T)*nxyz);
		new_device = false;
		new_host = false;
	}
	

	bool fft1DTo(Array<T>* dest){
		return internal_fft(dest, 1, FFT_FORWARD, &fft_plan_1D);
	}
	bool fft2DTo(Array<T>* dest){
		return internal_fft(dest, 2, FFT_FORWARD, &fft_plan_2D);
	}
	bool fft3DTo(Array<T>* dest){
		return internal_fft(dest, 3, FFT_FORWARD, &fft_plan_3D);
	}

	
	bool ifft1DTo(Array<T>* dest){
		return internal_fft(dest, 1, FFT_BACKWARD, &ifft_plan_1D);
	}
	bool ifft2DTo(Array<T>* dest){
		return internal_fft(dest, 2, FFT_BACKWARD, &ifft_plan_2D);
	}
	bool ifft3DTo(Array<T>* dest){
		return internal_fft(dest, 3, FFT_BACKWARD, &ifft_plan_3D);
	}

	
	bool areAllSameValue(T& v)
	{
		sync_hd();
		return arrayAreAllSameValue(d_data, nxyz, v);
	}

	void encode(buffer* b)
	{
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
			T* d = data();
			encodeInteger(flag, b);
			for(int i=0; i<nxyz; i++)
				luaT<T>::encode(d[i], b);
		}
	}
	int decode(buffer* b)
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
			T* d = data();
			for(int i=0; i<nxyz; i++)
				d[i] = luaT<T>::decode(b);
		}
		new_host = true;
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

	int lua_set(lua_State* L, int base_idx)
	{
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
		new_host = true;
		return 0;
	}
	
		
	int lua_addat(lua_State* L, int base_idx)
	{
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
			return luaL_error(L, "invalid site (%i, %i, %i)", c[0], c[1], c[2]);
		c[0] = xyz2idx(c[0], c[1], c[2]);
		
		plus_equal( data()[c[0]], luaT<T>::to(L, base_idx + offset) );
		new_host = true;
		return 0;
	}
	
	T get(int x=0, int y=0, int z=0)
	{
		return data()[xyz2idx(x,y,z)];
	}
	
	void set(int x, int y, int z, T v)
	{
		data()[xyz2idx(x,y,z)] = v;
		new_host = true;
	}
	
	int lua_get(lua_State* L, int base_idx)
	{
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
	
	void setAll(const T& v){arraySetAll(ddata(), v, nxyz);}
	void scaleAll(const T& v) {sync_hd(); arrayScaleAll(ddata(), v, nxyz);}
	void scaleAll_o(const T& v, const int offset, const int n) {sync_hd(); arrayScaleAll_o(ddata(), offset, v, n);}
	void addValue(const T& v) {sync_hd(); arrayAddAll(ddata(), v, nxyz);}
	
	
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
		arrayMultAll(dest->ddata(), src1->ddata(), src2->ddata(), dest->nxyz);
		return true;
	}

	static bool pairwiseDiff(Array<T>* dest, Array<T>* src1, Array<T>* src2)
	{
		if(!Array<T>::triplePrep(dest, src1, src2)) return false;
		arrayDiffAll(dest->ddata(), src1->ddata(), src2->ddata(), dest->nxyz);
		return true;
	}
	
	static bool pairwiseScaleAdd(Array<T>* dest, const T& s1, Array<T>* src1, const T& s2, Array<T>* src2)
	{
		if(!Array<T>::triplePrep(dest, src1, src2)) return false;
		arrayScaleAdd(dest->ddata(), s1, src1->ddata(), s2, src2->ddata(), dest->nxyz);
		return true;
	}

	static bool norm(Array<T>* dest, Array<T>* src)
	{
		if(!doublePrep(dest, src)) return false;
		arrayNormAll(dest->ddata(), src->ddata(), dest->nxyz);
		return true;
	}
	
	static T dot(Array<T>* a, Array<T>* b)
	{
		if(!doublePrep(a, b)) return luaT<T>::zero();
		T t;
		reduceMultSumAll(a->ddata(), b->ddata(), a->nxyz, t);
		return t;
	}

	T max(int& idx)
	{
		T v;
		reduceExtreme(ddata(), 1, nxyz, v, idx);
		return v;
	}
	T mean()
	{
		T v;
		reduceSumAll(ddata(), nxyz, v);
		divide_real<T>(v, v, nxyz);
		return v;
	}
	T min(int& idx)
	{
		T v;
		reduceExtreme(ddata(),-1, nxyz, v, idx);
		return v;
	}

	T sum()
	{
		sync_hd(); 
		T v;
		reduceSumAll(ddata(), nxyz, v);
		return v;
	}

	T diffSum(Array<T>* other)
	{
		sync_hd(); 
		T v;
		reduceDiffSumAll(ddata(), other->ddata(), nxyz, v);
		return v;
	}
	
	void copyFrom(Array<T>* other)
	{
		sync_hd();
		memcpy_d2d(ddata(), other->ddata(), sizeof(T)*nxyz);
		other->new_device = true;
		other->new_host = false;
	}
	
	void zero()
	{
		arraySetAll(ddata(),  luaT<T>::zero(), nxyz);
		new_device = true;
		new_host = false;
	}
	
	Array<T>& operator+=(Array<T> &rhs)
	{
		arraySumAll(ddata(), ddata(), rhs.ddata(), nxyz);
		return *this;
	}
	T& operator[](int index){
		return data()[index];
	}
};

#ifdef WIN32
 #ifndef DUMMYWINDOWS_ARRYA_INSTANTIATION
  #define DUMMYWINDOWS_ARRYA_INSTANTIATION
  //forcing instantiation so they get exported
  template class Array<doubleComplex>;
  template class Array<double>;
  template class Array<floatComplex>;
  template class Array<float>;
  template class Array<int>;
 #endif
#endif

typedef Array<doubleComplex> dcArray;
typedef Array<floatComplex>  fcArray;
typedef Array<double>         dArray;
typedef Array<float>          fArray;
typedef Array<int>            iArray;




ARRAYCUDA_API dcArray* getWSdcArray(int nx, int ny, int nz, long level);
ARRAYCUDA_API fcArray* getWSfcArray(int nx, int ny, int nz, long level);
ARRAYCUDA_API dArray* getWSdArray(int nx, int ny, int nz, long level);
ARRAYCUDA_API fArray* getWSfArray(int nx, int ny, int nz, long level);
ARRAYCUDA_API iArray* getWSiArray(int nx, int ny, int nz, long level);



#endif //#ifndef ARRAYCORECUDA

