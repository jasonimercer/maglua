#include "array.h"
#include "memory.hpp"





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
static const luaL_Reg* get_base_methods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	static const luaL_Reg _m[] =
	{
		{"setAll",  l_setAll<T>},
		{"zero",    l_zero<T>},
		{"sameSize",l_sameSize<T>},
		{"nx",      l_get_nx<T>},
		{"ny",      l_get_ny<T>},
		{"nz",      l_get_nz<T>},
		{"get",     l_get<T>},
		{"set",     l_set<T>},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}


template<typename T>
int Array<T>::help(lua_State* L)
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
	if(func == &(l_sameSize<T>))
	{
		lua_pushstring(L, "Test if a given array has the same dimensions");
		lua_pushstring(L, "1 Array");
		lua_pushstring(L, "1 Boolean: True if sizes match");
		return 3;
	}
	if(func == &(l_setAll<T>))
	{
		lua_pushstring(L, "Set all values in the array to the given value");
		lua_pushstring(L, "1 Value: The new value for all element entries");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == &(l_zero<T>))
	{
		lua_pushstring(L, "Set all values in the array 0");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == &(l_get_nx<T>))
	{
		lua_pushstring(L, "Return the size of the X dimension");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size fo the X dimension");
		return 3;
	}
	if(func == &(l_get_ny<T>))
	{
		lua_pushstring(L, "Return the size of the Y dimension");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size fo the Y dimension");
		return 3;
	}
	if(func == &(l_get_nz<T>))
	{
		lua_pushstring(L, "Return the size of the Z dimension");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size fo the Z dimension");
		return 3;
	}


	return LuaBaseObject::help(L);
}








static void* fft_ws = 0;
static int fft_ws_size = 0;

static void ensure_ws(const int ws_size)
{
	//printf("ws_size %i\n", ws_size);
	if(ws_size > fft_ws_size)
	{
		fft_ws_size = ws_size;
		if(fft_ws)
			free_device(fft_ws);
		malloc_device(&fft_ws, ws_size);
	}
}

template<typename T>
static int l_fft1D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	ensure_ws( sizeof(T) * a->nx * a->ny * a->nz );
	a->fft1DTo(b, (T*)fft_ws);
	return 0;
}
template<typename T>
static int l_fft2D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	ensure_ws( sizeof(T) * a->nx * a->ny * a->nz );
	a->fft2DTo(b, (T*)fft_ws);
	return 0;
}
template<typename T>
static int l_fft3D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	ensure_ws( sizeof(T) * a->nx * a->ny * a->nz );
	a->fft3DTo(b, (T*)fft_ws);
	return 0;
}



template<typename T>
static int l_ifft1D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	ensure_ws( sizeof(T) * a->nx * a->ny * a->nz );
	a->ifft1DTo(b, (T*)fft_ws);
	return 0;
}
template<typename T>
static int l_ifft2D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	ensure_ws( sizeof(T) * a->nx * a->ny * a->nz );
	a->ifft2DTo(b, (T*)fft_ws);
	return 0;
}
template<typename T>
static int l_ifft3D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	ensure_ws( sizeof(T) * a->nx * a->ny * a->nz );
	a->ifft3DTo(b, (T*)fft_ws);
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
			c[i] = lua_tonumber(L, i+1);
	}

	for(int i=0; i<3; i++)
		if(c[i] < 1) c[i] = 1;
	
	a->setSize(c[0], c[1], c[2]);
	return 0;
}



template <typename T>
const luaL_Reg* Array<T>::luaMethods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	merge_luaL_Reg(m, get_base_methods<T>());
	m[127].name = (char*)1;
	return m;
}

//special cases for complex datatypes (fft):
template <>
const luaL_Reg* Array<doubleComplex>::luaMethods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	merge_luaL_Reg(m, get_base_methods<doubleComplex>());
	merge_luaL_Reg(m, get_fft_methods<doubleComplex>());
	m[127].name = (char*)1;
	return m;
}
template <>
const luaL_Reg* Array<floatComplex>::luaMethods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	merge_luaL_Reg(m, get_base_methods<floatComplex>());
	merge_luaL_Reg(m, get_fft_methods<floatComplex>());
	m[127].name = (char*)1;
	return m;
}



template <typename T>
int Array<T>::luaInit(lua_State* L)
{
	return l_init<T>(this, L);
}

// template <typename T>
// int Array<T>::help(lua_State* L)
// {
// 	return 0;
// }




extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
        
ARRAYCUDA_API int lib_register(lua_State* L);
ARRAYCUDA_API int lib_deps(lua_State* L);
ARRAYCUDA_API int lib_version(lua_State* L);
ARRAYCUDA_API const char* lib_name(lua_State* L);
ARRAYCUDA_API int lib_main(lua_State* L);
}
#include "info.h"

ARRAYCUDA_API int lib_register(lua_State* L)
{
	luaT_register<dArray>(L);
	luaT_register<fArray>(L);
 	luaT_register<iArray>(L);
	luaT_register<dcArray>(L);
	luaT_register<fcArray>(L);

	return 0;
}

ARRAYCUDA_API int lib_version(lua_State* L)
{
	return __revi;
}

ARRAYCUDA_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Array-Cuda";
#else
	return "Array-Cuda-Debug";
#endif
}
