#include "array_core_cpu.h"
#include "array.h"


template<typename T>
static int l_sameSize(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	LUA_PREAMBLE(Array<T>, b, 2);
	lua_pushboolean(L, a->sameSize(b));
	return 1;
}

template<typename T>
static int l_get_nx(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	lua_pushinteger(L, a->nx);
	return 1;
}
template<typename T>
static int l_get_ny(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	lua_pushinteger(L, a->ny);
	return 1;
}
template<typename T>
static int l_get_nz(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	lua_pushinteger(L, a->nz);
	return 1;
}
template<typename T>
static int l_get(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	return a->lua_get(L, 2);
}
template<typename T>
static int l_set(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	return a->lua_set(L, 2);
}
template<typename T>
static int l_addat(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	return a->lua_addat(L, 2);
}


template<typename T>
static int l_setAll(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	T t = luaT<T>::to(L, 2);
	a->setAll(t);
	return 0;
}
template<typename T>
static int l_zero(lua_State* L)
{
 	LUA_PREAMBLE(Array<T>, a, 1);
	T t = luaT<T>::zero();
	a->setAll(t);
	return 0;
}
template<typename T>
static int l_pwm(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	LUA_PREAMBLE(Array<T>, b, 2);
	LUA_PREAMBLE(Array<T>, c, 3);
	
	Array<T>::pairwiseMult(c, a, b);
	return 0;
}
template<typename T>
static int l_dot(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	LUA_PREAMBLE(Array<T>, b, 2);
	
	T t = Array<T>::dot(a,b);
	luaT<T>::push(L, t);
	return 1;
}

template<typename T>
static int l_min(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	int idx;
	T t = a->min(idx);

	luaT<T>::push(L, t);
	lua_pushinteger(L, idx+1);
	return luaT<T>::elements() + 1;
}

template<typename T>
static int l_max(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	int idx;
	T t = a->max(idx);

	luaT<T>::push(L, t);
	lua_pushinteger(L, idx+1);
	return luaT<T>::elements() + 1;
}

template<typename T>
static int l_mean(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	T t = a->mean();
	luaT<T>::push(L, t);
	return luaT<T>::elements();
}



template<typename T>
static const luaL_Reg* get_base_methods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	static const luaL_Reg _m[] =
	{
		{"setAll",  l_setAll<T>},
		{"pairwiseMultiply",     l_pwm<T>},
		{"dot",     l_dot<T>},
		{"zero",    l_zero<T>},
		{"sameSize",l_sameSize< T >},
		{"nx",      l_get_nx<T>},
		{"ny",      l_get_ny<T>},
		{"nz",      l_get_nz<T>},
		{"get",     l_get<T>},
		{"set",     l_set<T>},
		{"addAt",   l_addat<T>},
		{"min",     l_min<T>},
		{"max",     l_max<T>},
		{"mean",    l_mean<T>},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}


template<typename T>
static int Array_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Class for 3D Data Arrays");
		lua_pushstring(L, "0 to 3 Integers: Length of each dimension X, Y and Z. Default values are 1.");
		lua_pushstring(L, ""); //output, empty
		return 3;
	}

	if(!lua_iscfunction(L, 1))
	{
		return luaL_error(L, "help expects zero arguments or 1 function.");
	}


	lua_CFunction func = lua_tocfunction(L, 1);
	
	lua_CFunction f01 = l_sameSize<T>; //THIS IS BULLSHIT! Doing this so that icc 11 can compile cleanly. 
	if(func == f01)
	{
		lua_pushstring(L, "Test if a given array has the same dimensions");
		lua_pushstring(L, "1 Array");
		lua_pushstring(L, "1 Boolean: True if sizes match");
		return 3;
	}
	lua_CFunction f02 = l_setAll<T>;
	if(func == f02)
	{
		lua_pushstring(L, "Set all values in the array to the given value");
		lua_pushstring(L, "1 Value: The new value for all element entries");
		lua_pushstring(L, "");
		return 3;
	}
	lua_CFunction f03 = l_pwm<T>;
	if(func == f03)
	{
		lua_pushstring(L, "Multiply each data in this array with the data in another storing in a destination array");
		lua_pushstring(L, "2 Arrays: The pairwise scaling array and the destination array");
		lua_pushstring(L, "");
		return 3;
	}
	lua_CFunction f04 = l_dot<T>;
	if(func == f04)
	{
		lua_pushstring(L, "Compute the dot product of the current array and another of equal size and type");
		lua_pushstring(L, "1 Arrays: other array");
		lua_pushstring(L, "1 Value: result of dot product");
		return 3;
	}
	
	lua_CFunction f05 = l_zero<T>;
	if(func == f05)
	{
		lua_pushstring(L, "Set all values in the array 0");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}
	lua_CFunction f06 = l_get_nx<T>;
	if(func == f06)
	{
		lua_pushstring(L, "Return the size of the X dimension");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size fo the X dimension");
		return 3;
	}
	lua_CFunction f07 = l_get_ny<T>;
	if(func == f07)
	{
		lua_pushstring(L, "Return the size of the Y dimension");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size fo the Y dimension");
		return 3;
	}
	lua_CFunction f08 = l_get_nz<T>;
	if(func == f08)
	{
		lua_pushstring(L, "Return the size of the Z dimension");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size fo the Z dimension");
		return 3;
	}

	lua_CFunction f09 = l_get<T>;
	if(func == f09)
	{
		lua_pushstring(L, "Get an element from the array");
		lua_pushstring(L, "1, 2 or 3 integers (or 1 table): indices(XYZ) of the element to fetch default values are 1");
		lua_pushstring(L, "1 value");
		return 3;
	}
	lua_CFunction f10 = l_set<T>;
	if(func == f10)
	{
		lua_pushstring(L, "Set an element of the array");
		lua_pushstring(L, "1, 2 or 3 integers (or 1 table), 1 value: indices(XYZ) of the element to set, default values are 1. Last argument is the new value");
		lua_pushstring(L, "");
		return 3;
	}
	lua_CFunction f11 = l_addat<T>;
	if(func == f11)
	{
		lua_pushstring(L, "Add a value to an element of the array");
		lua_pushstring(L, "1, 2 or 3 integers (or 1 table), 1 value: indices(XYZ) of the element to modify, default values are 1. Last argument is the value to add");
		lua_pushstring(L, "");
		return 3;
	}
	lua_CFunction f12 = l_min<T>;
	if(func == f12)
	{
		lua_pushstring(L, "Find minimum value and corresponding index");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 value and 1 integer");
		return 3;
	}
	lua_CFunction f13 = l_max<T>;
	if(func == f13)
	{
		lua_pushstring(L, "Find maximum value and corresponding index");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 value and 1 integer");
		return 3;
	}
	lua_CFunction f14 = l_mean<T>;
	if(func == f14)
	{
		lua_pushstring(L, "Find mean of array");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 value");
		return 3;
	}
	return LuaBaseObject::help(L);
}



int array_help_specialization_int(lua_State* L)
{
	return Array_help<int>(L);
}
int array_help_specialization_float(lua_State* L)
{
	return Array_help<float>(L);
}
int array_help_specialization_double(lua_State* L)
{
	return Array_help<double>(L);
}
int array_help_specialization_floatComplex(lua_State* L)
{
	return Array_help<floatComplex>(L);
}
int array_help_specialization_doubleComplex(lua_State* L)
{
	return Array_help<doubleComplex>(L);
}





template<typename T>
static int l_fft1D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	a->fft1DTo(b, (T*)0);
	return 0;
}
template<typename T>
static int l_fft2D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	a->fft2DTo(b, (T*)0);
	return 0;
}
template<typename T>
static int l_fft3D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	a->fft3DTo(b, (T*)0);
	return 0;
}



template<typename T>
static int l_ifft1D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	a->ifft1DTo(b, (T*)0);
	return 0;
}
template<typename T>
static int l_ifft2D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	a->ifft2DTo(b, (T*)0);
	return 0;
}
template<typename T>
static int l_ifft3D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	a->ifft3DTo(b, (T*)0);
	return 0;
}


template<typename T>
static const luaL_Reg* get_fft_methods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	static const luaL_Reg _m[] =
	{
		{"fft1D",   l_fft1D<T>},
		{"fft2D",   l_fft2D<T>},
		{"fft3D",   l_fft3D<T>},

		{"ifft1D",  l_ifft1D<T>},
		{"ifft2D",  l_ifft2D<T>},
		{"ifft3D",  l_ifft3D<T>},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}











template<typename T>
static int l_init( Array<T>* a, lua_State* L)
{
	int c[3] = {1,1,1};

	if(lua_istable(L, 1))
	{
		for(int i=0; i<3; i++)
		{
			lua_pushinteger(L, i+1);
			lua_gettable(L, 1);
			c[i] = lua_tointeger(L, -1);
			lua_pop(L, 1);
		}
	}
	else
	{
		for(int i=0; i<3; i++)
			if(lua_isnumber(L, i+1))
				c[i] = lua_tonumber(L, i+1);
	}

	for(int i=0; i<3; i++)
		if(c[i] < 0) c[i] = 0;
	
	a->setSize(c[0], c[1], c[2]);
	return 0;
}





template <typename T>
const luaL_Reg* Array_luaMethods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	merge_luaL_Reg(m, get_base_methods<T>());
	m[127].name = (char*)1;
	return m;
}

//special cases for complex datatypes (fft):
template <>
const luaL_Reg* Array_luaMethods<doubleComplex>()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	merge_luaL_Reg(m, get_base_methods<doubleComplex>());
	merge_luaL_Reg(m, get_fft_methods<doubleComplex>());
	m[127].name = (char*)1;
	return m;
}

template <>
const luaL_Reg* Array_luaMethods<floatComplex>()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	merge_luaL_Reg(m, get_base_methods<floatComplex>());
	merge_luaL_Reg(m, get_fft_methods<floatComplex>());
	m[127].name = (char*)1;
	return m;
}


const luaL_Reg* array_luamethods_specialization_int()
{
  return Array_luaMethods<int>();
}
const luaL_Reg* array_luamethods_specialization_float()
{
  return Array_luaMethods<float>();
}
const luaL_Reg* array_luamethods_specialization_double()
{
  return Array_luaMethods<double>();
}

const luaL_Reg* array_luamethods_specialization_doubleComplex()
{
  return Array_luaMethods<doubleComplex>();
}
const luaL_Reg* array_luamethods_specialization_floatComplex()
{
  return Array_luaMethods<floatComplex>();
}



int array_luainit_specialization_int(Array<int>* that, lua_State* L)
{
  return l_init<int>(that, L);
}
int array_luainit_specialization_float(Array<float>* that, lua_State* L)
{
  return l_init<float>(that, L);
}
int array_luainit_specialization_double(Array<double>* that, lua_State* L)
{
  return l_init<double>(that, L);
}
int array_luainit_specialization_floatComplex(Array<floatComplex>* that, lua_State* L)
{
  return l_init<floatComplex>(that, L);
}
int array_luainit_specialization_doubleComplex(Array<doubleComplex>* that, lua_State* L)
{
  return l_init<doubleComplex>(that, L);
}

/*
template <typename T>
int Array<T>::luaInit(lua_State* L)
{
	return l_init<T>(this, L);
}
*/

// template <typename T>
// int Array<T>::help(lua_State* L)
// {
// 	return 0;
// }






// the level argument below prevents WSs from overlapping. This is useful for multi-level 
// operations that all use WSs: example long range interaction. FFTs at lowest level with a WS acting as an accumulator
template<typename T>
 Array<T>* getWSArray(int nx, int ny, int nz, long level)
{
	T* m;
	T* h;
	getWSMem(&m, sizeof(T)*nx*ny*nz, level);
	//printf("New WS Array %p %p\n", d, h);
	Array<T>* a =  new Array<T>(nx,ny,nz,m);

	//printf("ddata = %p\n", a->ddata());

	return a;
}

ARRAY_API dcArray* getWSdcArray(int nx, int ny, int nz, long level)
{
	return getWSArray<doubleComplex>(nx,ny,nz,level);
}
ARRAY_API fcArray* getWSfcArray(int nx, int ny, int nz, long level)
{
	return getWSArray<floatComplex>(nx,ny,nz,level);
}
ARRAY_API dArray* getWSdArray(int nx, int ny, int nz, long level)
{
	return getWSArray<double>(nx,ny,nz,level);
}
ARRAY_API fArray* getWSfArray(int nx, int ny, int nz, long level)
{
	return getWSArray<float>(nx,ny,nz,level);
}
ARRAY_API iArray* getWSiArray(int nx, int ny, int nz, long level)
{
	return getWSArray<int>(nx,ny,nz,level);
}

















#ifdef WIN32
 #ifdef ARRAY_EXPORTS
  #define ARRAY_API __declspec(dllexport)
 #else
  #define ARRAY_API __declspec(dllimport)
 #endif
#else
 #define ARRAY_API 
#endif

extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
        
ARRAY_API int lib_register(lua_State* L);
ARRAY_API int lib_deps(lua_State* L);
ARRAY_API int lib_version(lua_State* L);
ARRAY_API const char* lib_name(lua_State* L);
ARRAY_API int lib_main(lua_State* L);
}
#include "info.h"

ARRAY_API int lib_register(lua_State* L)
{
	dArray foo1;

#ifdef DOUBLE_ARRAY
	luaT_register< Array<double> >(L);
#endif
#ifdef SINGLE_ARRAY
	luaT_register<fArray>(L);
#endif
	luaT_register<iArray>(L);
#ifdef DOUBLE_ARRAY
	luaT_register<dcArray>(L);
#endif
#ifdef SINGLE_ARRAY
	luaT_register<fcArray>(L);
#endif

	return 0;
}

ARRAY_API int lib_version(lua_State* L)
{
	return __revi;
}

ARRAY_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Array";
#else
	return "Array-Debug";
#endif
}


