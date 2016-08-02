#include "luabaseobject.h"

#ifdef _LAPACK
#ifndef ARRAY_CORE_MATRIX_CPU_LAPACK
#define ARRAY_CORE_MATRIX_CPU_LAPACK

#include "array.h"
#include <string>


int l_mateigen(lua_State* L);
int l_matcond(lua_State* L);
int l_matinverse(lua_State* L);
int l_matmul(lua_State* L);
int l_matsvd(lua_State* L);
int l_matlinearsystem(lua_State* L);

int l_mat_lapack_help(lua_State* L);


bool array_matlinearsystem(dArray* A, dArray* x, dArray* b, std::string& err_msg);


#endif
#endif
