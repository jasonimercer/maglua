#ifndef ARRAY_MEMORY_H
#define ARRAY_MEMORY_H


#include "luabaseobject.h"
#include "array_ops.h"

ARRAY_API void  registerWS();
ARRAY_API void  unregisterWS();
ARRAY_API int l_ws_info(lua_State* L);

// the level argument below prevents WSs from overlapping. This is useful for multi-level 
// operations that all use WSs: example long range interaction. FFTs at lowest level with
// a WS acting as an accumulator
ARRAY_API bool getWSMem_(void** ptr, size_t size, long level);
#define getWSMem(p1,s1,level) getWSMem_((void**)p1, s1, level)

#endif
