#include "luabaseobject.h"

#ifdef _LAPACK
#define ARRAY_CORE_MATRIX_CPU_LAPACK

int l_mateigen(lua_State* L);
int l_matcond(lua_State* L);
int l_matinverse(lua_State* L);
int l_matmul(lua_State* L);
int l_matsvd(lua_State* L);
int l_matlinearsystem(lua_State* L);

int l_mat_lapack_help(lua_State* L);
#endif
