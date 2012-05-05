#include "array_core_cpu.h"
#include "luabaseobject.h"

#ifndef ARRAY_H
#define ARRAY_H

template<typename T>
inline const char* array_lua_name() {return "Array.Unnamed";}

template<>inline const char* array_lua_name<int>() {return "Array.Integer";}
template<>inline const char* array_lua_name<float>() {return "Array.Float";}
template<>inline const char* array_lua_name<double>() {return "Array.Double";}
template<>inline const char* array_lua_name<floatComplex>() {return "Array.FloatComplex";}
template<>inline const char* array_lua_name<doubleComplex>() {return "Array.DoubleComplex";}

template<typename T>
class Array : public ArrayCore<T>, public LuaBaseObject	
{
public:
	Array(int nx=4, int ny=4, int nz=1) : ArrayCore<T>(nx, ny, nz), LuaBaseObject(hash32(lineage(0))) {}
	LINEAGE1(array_lua_name<T>())
	static const luaL_Reg* luaMethods(); 
	virtual int luaInit(lua_State* L); 
	static int help(lua_State* L); 	
};

typedef Array<doubleComplex> dcArray;
typedef Array<floatComplex>  fcArray;
typedef Array<double>         dArray;
typedef Array<float>          fArray;
typedef Array<int>            iArray;

#endif
