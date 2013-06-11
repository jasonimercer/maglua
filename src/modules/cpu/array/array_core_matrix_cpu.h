#include "array_core_cpu.h"

#ifdef _MATRIX
#define ARRAY_CORE_MATRIX_CPU

int Array_help_matrix_int(lua_State* L);
int Array_help_matrix_float(lua_State* L);
int Array_help_matrix_double(lua_State* L);
int Array_help_matrix_floatComplex(lua_State* L);
int Array_help_matrix_doubleComplex(lua_State* L);


const luaL_Reg* get_base_methods_matrix_int();
const luaL_Reg* get_base_methods_matrix_float();
const luaL_Reg* get_base_methods_matrix_double();
const luaL_Reg* get_base_methods_matrix_floatComplex();
const luaL_Reg* get_base_methods_matrix_doubleComplex();

#endif
