#include <cuda.h>
#include <cuda_runtime.h>
#include "luabaseobject.h"
#include <cuComplex.h>

#ifndef LUA_TYPE_TRANSLATE
#define LUA_TYPE_TRANSLATE


#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef ARRAYCUDA_EXPORTS
  #define ARRAYCUDA_API __declspec(dllexport)
 #else
  #define ARRAYCUDA_API __declspec(dllimport)
 #endif
#else
 #define ARRAYCUDA_API 
#endif

// These static methods define Lua and buffer interaction for different data types

template<typename T>
class ARRAYCUDA_API luaT
{
public:
	static int elements() {return 0;}
	static int push(lua_State* L, const T& v){return elements();}
	static T to(lua_State* L, int idx){return 0;}
	static int is(lua_State* L, int idx){return 0;}
	static	void encode(const T& v, buffer* b){}
	static	T decode(buffer* b){return 0;}
	static	T zero() {return 0;}
	static	T one() {return 1;}
	static	T neg_one() {return -1;}
	static bool lt(const T& a, const T& b) {return false;}
};

template<>
class ARRAYCUDA_API luaT<double>{
public:
	static int elements() {return 1;}
	static int push(lua_State* L, const double& v){lua_pushnumber(L, v);return elements();}
	static double to(lua_State* L, int idx){return lua_tonumber(L, idx);}
	static int is(lua_State* L, int idx){return lua_isnumber(L, idx);}
	static	void encode(const double& v, buffer* b){	encodeDouble(v, b);	}
	static	double decode(buffer* b){return decodeDouble(b);}
	static	double zero() {return 0;}
	static	double one() {return 1;}
	static	double neg_one() {return -1;}
	static bool lt(const double& a, const double& b) {return a<b;}
};

template<>
class ARRAYCUDA_API luaT<float>{
public:
	static int elements() {return 1;}
	static int push(lua_State* L, const float& v){lua_pushnumber(L, v);return elements();}
	static float to(lua_State* L, int idx){return (float)lua_tonumber(L, idx);}
	static int is(lua_State* L, int idx){return lua_isnumber(L, idx);}
	static	void encode(const float& v, buffer* b){	encodeDouble(v, b);	}
	static	float decode(buffer* b){return (float)decodeDouble(b);}
	static	float zero() {return 0;}
	static	float one() {return 1;}
	static	float neg_one() {return -1;}
	static bool lt(const float& a, const float& b) {return a<b;}
};

template<>
class ARRAYCUDA_API luaT<int>{
public:
	static int elements() {return 1;}
	static int push(lua_State* L, const int& v){lua_pushnumber(L, v);return elements();}
	static int to(lua_State* L, int idx){return lua_tointeger(L, idx);}
	static int is(lua_State* L, int idx){return lua_isnumber(L, idx);}
	static	void encode(const int& v, buffer* b){	encodeInteger(v, b);	}
	static	int decode(buffer* b){return decodeInteger(b);}
	static	int zero() {return 0;}
	static	int one() {return 1;}
	static	int neg_one() {return -1;}
	static bool lt(const int& a, const int& b) {return a<b;}
};

template<>
class ARRAYCUDA_API luaT<cuDoubleComplex>{
public:
	static int elements() {return 2;}
	static int push(lua_State* L, const cuDoubleComplex& v){lua_pushnumber(L, v.x);lua_pushnumber(L, v.y);return 2;}
	static cuDoubleComplex to(lua_State* L, int idx){double a = lua_tonumber(L, idx); double b = lua_tonumber(L, idx+1); return make_cuDoubleComplex(a,b);}
	static int is(lua_State* L, int idx){return lua_isnumber(L, idx) && lua_isnumber(L, idx+1);}
	static	void encode(const cuDoubleComplex& v, buffer* b){	encodeDouble(v.x, b);encodeDouble(v.y, b);}
	static	cuDoubleComplex decode(buffer* b){return make_cuDoubleComplex(decodeDouble(b), decodeDouble(b));}
	static	cuDoubleComplex zero() {return make_cuDoubleComplex(0,0);}
	static	cuDoubleComplex one() {return make_cuDoubleComplex(1,0);}
	static	cuDoubleComplex neg_one() {return make_cuDoubleComplex(-1,0);}
};

template<>
class ARRAYCUDA_API luaT<floatComplex>{
public:
	static int elements() {return 2;}
	static int push(lua_State* L, const cuFloatComplex& v){lua_pushnumber(L, v.x);lua_pushnumber(L, v.y);return 2;}
	static cuFloatComplex to(lua_State* L, int idx){float a = lua_tonumber(L, idx); float b = lua_tonumber(L, idx+1); return make_cuFloatComplex(a,b);}
	static int is(lua_State* L, int idx){return lua_isnumber(L, idx) && lua_isnumber(L, idx+1);}
	static	void encode(const cuFloatComplex& v, buffer* b){	encodeDouble(v.x, b);encodeDouble(v.y, b);}
	static	cuFloatComplex decode(buffer* b){return make_cuFloatComplex(decodeDouble(b), decodeDouble(b));}
	static	cuFloatComplex zero() {return make_cuFloatComplex(0,0);}
	static	cuFloatComplex one() {return make_cuFloatComplex(1,0);}
	static	cuFloatComplex neg_one() {return make_cuFloatComplex(-1,0);}
};

#endif
