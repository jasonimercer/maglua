#include "memory.h"
#include <fftw3.h>

#if 0
ARRAY_API void  registerWS() {};
ARRAY_API void  unregisterWS() {};
ARRAY_API bool getWSMem_(void** ptr, size_t size, long level) {*ptr = 0; return true;};
#endif 

#if 1
typedef struct work_space_memory
{
	int refcount;
	void* memory[64];
	size_t size[64];
	int slot_refcount[64];
	long level[64];
	
} work_space_memory;

static work_space_memory WS_MEM = {0};

ARRAY_API void  registerWS()
{
	if(WS_MEM.refcount == 0) //initialize
	{
		for(int i=0; i<64; i++)
		{
			WS_MEM.memory[i] = 0;
			WS_MEM.size[i] = 0;
			WS_MEM.slot_refcount[i] = 0;
			WS_MEM.level[i] = -1000; //dumb
		}
	}
	
	WS_MEM.refcount++;
}

ARRAY_API int l_ws_info(lua_State* L)
{
	lua_newtable(L);
	if(WS_MEM.refcount == 0)
		return 1;

	int p = 0;

	for(int i=0; i<64; i++)
	{
		if(WS_MEM.level[i] != -1000)
		{
			p++;
			lua_pushinteger(L, p);

			lua_newtable(L);
			lua_pushstring(L, "size");
			lua_pushinteger(L, WS_MEM.size[i]);
			lua_settable(L, -3);
			lua_pushstring(L, "hash");
			lua_pushinteger(L, WS_MEM.level[i]);
			lua_settable(L, -3);

			lua_settable(L, -3);
		}

	}

	return 1;
}

ARRAY_API void unregisterWS()
{
	WS_MEM.refcount--;
	if(WS_MEM.refcount == 0)
	{
		for(int i=0; i<64; i++)
		{
			if(WS_MEM.memory[i])
				fftw_free(WS_MEM.memory[i]);
			WS_MEM.memory[i] = 0;
			WS_MEM.size[i] = 0;
			WS_MEM.level[i] = -1000; //dumb
		}
	}

}

// multiple operations can use the same workspace so you had better make sure 
// the workspace isn't storing something important: Keep ws usage contained
ARRAY_API bool getWSMem_(void** ptr, size_t size, long level)
{
	for(int i=0; i<64; i++)
	{
		if(WS_MEM.level[i] == -1000 || WS_MEM.level[i] == level)
		{
			if(WS_MEM.size[i] == 0)
			{
				WS_MEM.memory[i] = fftw_malloc(size);
				WS_MEM.size[i] = size;
			}
			
			if(WS_MEM.size[i] >= size)
			{
				WS_MEM.slot_refcount[i]++;
				*ptr = WS_MEM.memory[i];
				WS_MEM.level[i] = level;
				return true;
			}
		}
	}
	
	char buffer[1024];
	
#define log_print(a) fprintf(stderr, "%s", a)
	
	snprintf(buffer, 1024, "Failed to allocate memory for data of size %li (level = %li)\n", size, level);
	log_print(buffer);
	snprintf(buffer, 1024, "Here is the current list:\n");
	log_print(buffer);
	for(int i=0; i<64; i++)
	{
		snprintf(buffer, 1024, "% 3i  size: %8li  level: %8li\n", i, WS_MEM.size[i], WS_MEM.level[i]);
		log_print(buffer);
	}
	
	*ptr = 0;
	return false;
}
#endif
