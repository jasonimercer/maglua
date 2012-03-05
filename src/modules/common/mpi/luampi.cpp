/******************************************************************************
* Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#ifdef _MPI
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include "luampi.h"
#include "luamigrate.h"

extern "C" {
        #include <lua.h>
        #include <lualib.h>
        #include <lauxlib.h>
}


// GCC < 4.5 has buggy template pointers
// The following is to work around it by 
// making wrapper function
#ifdef __GNUC__
  #if __GNUC__ < 4
    #define BUGGY_TEMPLATE_POINTER
  #else
    #if __GNUC_MINOR__ < 5
      #define BUGGY_TEMPLATE_POINTER
    #endif
  #endif
#endif

#ifdef BUGGY_TEMPLATE_POINTER
#warning Using templated function pointer workaround for buggy gcc (gcc<4.5)
  //templated function address
  #define TFA(n,b) (&n ## _ ## b)
  //templated functiion addres wrapper
  #define TFAW(n) \
  static int n ## _0(lua_State* L) {return n  <0>(L);} \
  static int n ## _1(lua_State* L) {return n  <1>(L);}
#else
  #define TFA(n,b) (&n<b>)
  #define TFAW(n) 
#endif

#include "info.h"

#define NUMLUAVAR_TAG 100
#define    LUAVAR_TAG 200
#define   BUFSIZE_TAG 300
#define   GATHER_STEP 1000

inline int mpi_rank(MPI_Comm comm = MPI_COMM_WORLD)
{
	int rank;
	MPI_Comm_rank(comm, &rank);
	return rank;
}

inline int mpi_size(MPI_Comm comm = MPI_COMM_WORLD)
{
	int size;
	MPI_Comm_size(comm, &size);
	return size;
}



typedef struct mpi_comm_lua
{
	mpi_comm_lua(MPI_Comm c) {comm = c;}
	MPI_Comm comm;
    int refcount;
} mpi_comm_lua;

mpi_comm_lua* checkMPI_Comm(lua_State* L, int idx)
{
    mpi_comm_lua** pp = (mpi_comm_lua**)luaL_checkudata(L, idx, "MERCER.mpi_comm");
    luaL_argcheck(L, pp != NULL, 1, "`MPI_Comm' expected");
    return *pp;
}

void lua_pushMPI_Comm(lua_State* L, mpi_comm_lua* p)
{
    p->refcount++;
    mpi_comm_lua** pp = (mpi_comm_lua**)lua_newuserdata(L, sizeof(mpi_comm_lua**));
    *pp = p;
    luaL_getmetatable(L, "MERCER.mpi_comm");
    lua_setmetatable(L, -2);
}

static int l_MPI_Comm_gc(lua_State* L)
{
    mpi_comm_lua* p = checkMPI_Comm(L, 1);
    if(!p) return 0;

    p->refcount--;

    if(p->refcount <= 0)
    {
		delete p;
    }
	
    return 0;
}

static int l_MPI_Comm_tostring(lua_State* L)
{
	lua_pushstring(L, "MPI_Comm");
    return 1;
}



template<int base>
static MPI_Comm get_comm(lua_State* L)
{
	if(base == 0)
		return MPI_COMM_WORLD;
	else
	{
		mpi_comm_lua* p = checkMPI_Comm(L, 1);
		if(p)
			return p->comm;
	}
	return MPI_COMM_NULL;
}

template<int base>
static int l_mpi_comm_split(lua_State* L)
{
	MPI_Comm old = get_comm<base>(L);

	int colour = lua_tointeger(L, base+1);
	
	MPI_Comm new_comm;
	MPI_Comm_split(old, colour, 0, &new_comm);
	lua_pushMPI_Comm(L, new mpi_comm_lua(new_comm));

    return 1;
}
TFAW(l_mpi_comm_split)

template<int base>
static int l_cart_create(lua_State* L)
{
	MPI_Comm old = get_comm<base>(L);
	
	if(!lua_istable(L, base+1))
		return luaL_error(L, "First argument must be a table representing cartesian dimensions (max 10)");
	if(!lua_istable(L, base+2))
		return luaL_error(L, "Second argument must be a table of booleans indicating periodicity in that dimension (max 10)");
	
	int dims[10] = {0,0,0,0,0,0,0,0,0,0};
	int per[10] = {0,0,0,0,0,0,0,0,0,0};
	int ndims = 0;
	
	for(int i=0; i<10; i++)
	{
		lua_pushinteger(L, i+1);
		lua_gettable(L, base+1);
		dims[i] = lua_tointeger(L, -1);
		lua_pop(L, 1);
		lua_pushinteger(L, i+1);
		lua_gettable(L, base+2);
		per[i] = lua_toboolean(L, -1);		
		lua_pop(L, 1);
		
		if(dims[i] == 0)
			break;
		ndims++;
	}
	
	int reorder = lua_tointeger(L, base+3);
	
	if(ndims == 0)
		return luaL_error(L, "Zero dimensional grid not allowed");
	
	MPI_Comm new_comm;
	MPI_Cart_create(old, ndims, dims, per, reorder, &new_comm);

	lua_pushMPI_Comm(L, new mpi_comm_lua(new_comm));

    return 1;
}
TFAW(l_cart_create)

static int l_mpi_get_processor_name(lua_State* L)
{
	int  namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	
	MPI_Get_processor_name(processor_name, &namelen);
	
	lua_pushstring(L, processor_name);
	return 1;
}


template<int base>
static int l_mpi_get_size(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	lua_pushinteger(L, mpi_size(comm));
	return 1;
}
TFAW(l_mpi_get_size)


template<int base>
static int l_mpi_get_rank(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	lua_pushinteger(L, mpi_rank(comm)+1);
	return 1;
}
TFAW(l_mpi_get_rank)


template<int base>
static int l_mpi_barrier(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	MPI_Barrier(comm);
	return 0;
}
TFAW(l_mpi_barrier)


template<int base>
static int l_mpi_send(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	int dest = lua_tointeger(L, base+1) - 1; //lua is base 1
	
	if(dest < 0 || dest >= mpi_size(comm))
		return luaL_error(L, "Send destination (%d) is out of range.", dest+1); 
	
	int n = lua_gettop(L) - 1;
	int size;
	char* buf;
	
	MPI_Send(&n, 1, MPI_INT, dest, NUMLUAVAR_TAG, comm);
	
	for(int i=0; i<n; i++)
	{
		buf = exportLuaVariable(L, base+i+2, &size);
		
		MPI_Send(&size, 1, MPI_INT, dest, BUFSIZE_TAG+i, comm);
		MPI_Send(buf, size, MPI_CHAR, dest, LUAVAR_TAG+i, comm);
		free(buf);
	}
	return 0;
}
TFAW(l_mpi_send)


template<int base>
static int l_mpi_recv(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	int src = lua_tointeger(L, base+1) - 1; //lua is base 1
	int n;
	MPI_Status stat;

	if(src < 0 || src >= mpi_size(comm))
		return luaL_error(L, "Receive source (%d) is out of range.", src+1);

	char* buf = 0;
	int bufsize = 0;
	
	int reqBufSize;
	
	MPI_Recv(&n, 1, MPI_INT, src, NUMLUAVAR_TAG, comm, &stat);
	for(int i=0; i<n; i++)
	{
		MPI_Recv(&reqBufSize, 1, MPI_INT, src, BUFSIZE_TAG+i, comm, &stat);
		if(reqBufSize > bufsize)
		{
			buf = (char*)realloc(buf, reqBufSize);
			bufsize = reqBufSize;
		}
		
		MPI_Recv(buf, reqBufSize, MPI_CHAR, src, LUAVAR_TAG+i, comm, &stat);
		
		importLuaVariable(L, buf, reqBufSize);
	}
	
	if(buf)
		free(buf);
	return n;
}
TFAW(l_mpi_recv)


template<int base>
int l_mpi_gather(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	int r = mpi_rank(comm);
	int s = mpi_size(comm);
	int n = lua_gettop(L)-base;

	int root = lua_tointeger(L, base+1)-1;

	if(root < 0 || root >= s)
		return luaL_error(L, "invalid rank");

	int size = 0;

	int* rs = new int [s]; //remote_sizes
	for(int i=0; i<s; i++)
		rs[i] = 0;

	char* buf = exportLuaVariable(L, base+2, &size);

	lua_pop(L, n); //clear stack

	MPI_Gather(&size, 1, MPI_INT, rs, 1, MPI_INT, root, comm);

	int ms = 0; //max size
	if(r == root)
	{
		ms = rs[0]; //max size
		for(int i=1; i<s; i++)
		{
			if(rs[i] > ms)
				ms = rs[i];
		}
	}


	MPI_Bcast(&ms, 1, MPI_INT, root, comm);

	char* big_chunk = (char*) malloc( (sizeof(int)+ms) * s  );
	char* little_chunk = (char*) malloc(sizeof(int)+ms );

	bzero(big_chunk, (sizeof(int)+ms) * s);
	bzero(little_chunk, sizeof(int)+ms);

	memcpy(little_chunk, &size, sizeof(int));
	memcpy(little_chunk + sizeof(int), buf, size);

	MPI_Gather(little_chunk, sizeof(int)+ms, MPI_CHAR, big_chunk, sizeof(int)+ms, MPI_CHAR, root, comm);


	if(r == root)
	{
		lua_newtable(L);
		for(int i=0; i<s; i++)
		{
			lua_pushinteger(L, i+1);
			importLuaVariable(L, big_chunk + sizeof(int) + (sizeof(int) + ms) * i, rs[i]);
			lua_settable(L, -3);
		}
	}


	free(buf);
	delete [] rs;
	free(little_chunk);
	free(big_chunk);

	if(r == root)
		return 1;
	return 0;
}
TFAW(l_mpi_gather)


template<int base>
int l_mpi_bcast(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	int r = mpi_rank(comm);
	int s = mpi_size(comm);
	int n = lua_gettop(L)-base;

	int root = lua_tointeger(L, base+1)-1;

	if(root < 0 || root >= s)
		return luaL_error(L, "invalid rank");

	char* buf = 0;
	int size;
	if(root == r)
	{
		buf = exportLuaVariable(L, base+2, &size);
	}
	
	MPI_Bcast(&size, 1, MPI_INT, root, comm);

	if(root != r)
	{
		buf = (char*) malloc(size);
	}

	MPI_Bcast(buf, size, MPI_CHAR, root, comm);

	if(root == r)
	{
		lua_pushvalue(L, base+2);
	}
	else
	{
		importLuaVariable(L, buf, size);
	}
	free(buf);
	return 1;
}
TFAW(l_mpi_bcast)

static int l_mpi_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Exposes basic MPI functions");
		lua_pushstring(L, ""); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	if(lua_istable(L, 1))
	{
		return 0;
	}
	
	if(!lua_iscfunction(L, 1))
	{
		return luaL_error(L, "help expect zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if(func == l_mpi_get_processor_name)
	{
		lua_pushstring(L, "Returns the name of the processor as known by MPI.");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 String: Name");
		return 3;
	}	
	if(func == TFA(l_mpi_get_size,0) || func == TFA(l_mpi_get_size,1))
	{
		lua_pushstring(L, "Return the total number of proccesses in the global workgroup");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 Number: Number of processes");
		return 3;
	}	
	if(func == TFA(l_mpi_get_rank,0) || func == TFA(l_mpi_get_rank,1))
	{
		lua_pushstring(L, "The rank of the calling process");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 Number: Rank");
		return 3;
	}	
	if(func == TFA(l_mpi_send,0) || func == TFA(l_mpi_send,1))
	{
		lua_pushstring(L, "Send data to another process in the workgroup");
		lua_pushstring(L, "1 Number, ...: Index of remote process followed by zero or more variables"); 
		lua_pushstring(L, "");
		return 3;
	}	
	if(func == TFA(l_mpi_recv,0) || func == TFA(l_mpi_recv,1))
	{
		lua_pushstring(L, "Receive data from another process in the workgroup");
		lua_pushstring(L, "1 Number: Index of remote process sending the data"); 
		lua_pushstring(L, "...: zero or more pieces of data");
		return 3;
	}	
	if(func == TFA(l_mpi_barrier,0) || func == TFA(l_mpi_barrier,1))
	{
		lua_pushstring(L, "Syncronization barrier");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "");
		return 3;
	}

	if(func == TFA(l_mpi_gather,0) || func == TFA(l_mpi_gather,1))
	{
		lua_pushstring(L, "Gather data to a given rank. The return at the rank will be a table with each data as the value for source keys.");
		lua_pushstring(L, "1 Integer, 1 value: The value is what will be gathered, the Integer is the rank where the data will be gathered."); 
		lua_pushstring(L, "1 table or nil. The table will be returned at the given rank otherwise nil.");
		return 3;
	}
	
	if(func == TFA(l_mpi_bcast,0) || func == TFA(l_mpi_bcast,1))
	{
		lua_pushstring(L, "Broadcast data to all members of the workgrroup from the given rank.");
		lua_pushstring(L, "1 Integer, 1 value: The value is what will be broadcasted, the integer is the rank where the data will be from."); 
		lua_pushstring(L, "1 value. The broadcasted data.");
		return 3;
	}
	
	if(func == TFA(l_cart_create,0) || func == TFA(l_cart_create,1))
	{
		lua_pushstring(L, "Create a cartesian workgroup.");
		lua_pushstring(L, "2 Tables, 1 Boolean: The first table contains sizes of each dimension, the second is a table of booleans indicating if each dimension should be periodic. The last boolean argument states if the ranks may be reordered"); 
		lua_pushstring(L, "1 MPI_Comm: Optimized for cartesian communication");
		return 3;
	}
	
	if(func == TFA(l_mpi_comm_split,0) || func == TFA(l_mpi_comm_split,1))
	{
		lua_pushstring(L, "Split a workgroup into sub groups by common colours.");
		lua_pushstring(L, "1 Integer: The colour for the MPI_Comm_split function. Processes with common colours will be put into common sub-workgroups."); 
		lua_pushstring(L, "1 MPI_Comm: Sub-Workgroup");
		return 3;
	}

	return 0;
}

static int l_mpi_metatable(lua_State* L)
{
	lua_newtable(L);
	return 1;
}


#if 0
int l_mpirequest_gc(lua_State* L)
{
	mpi_req* io = checkRequest(L, 1);
	if(!io) return 0;

	io->refcount--;
	if(io->refcount == 0)
		delete io;

	return 0;
}
#endif

#define add(name, func) \
	lua_pushstring(L, name); \
	lua_pushcfunction(L, func); \
	lua_settable(L, -3); 

void registerMPI(lua_State* L)
{
    luaL_newmetatable(L, "MERCER.mpi_comm");
    lua_pushstring(L, "__index");
    lua_pushvalue(L, -2);  /* pushes the metatable */
    lua_settable(L, -3);  /* metatable.__index = metatable */
	add("get_size",           TFA(l_mpi_get_size,1)   );
	add("get_rank",           TFA(l_mpi_get_rank,1)   );
	add("send",               TFA(l_mpi_send,1)       );
	add("recv",               TFA(l_mpi_recv,1)       );
	add("barrier",            TFA(l_mpi_barrier,1)    );
	add("gather",             TFA(l_mpi_gather,1)     );
	add("bcast",              TFA(l_mpi_bcast,1)      );
	add("cart_create",        TFA(l_cart_create,1)    );
	add("comm_split",         TFA(l_mpi_comm_split,1) );
	add("__gc",               l_MPI_Comm_gc       );
	add("__tostring",         l_MPI_Comm_tostring );
	lua_pop(L,1); //metatable is registered
	

	lua_newtable(L);
	add("get_processor_name", l_mpi_get_processor_name);
	add("get_size",           TFA(l_mpi_get_size,0)          );
	add("get_rank",           TFA(l_mpi_get_rank,0)          );
	add("send",               TFA(l_mpi_send,0)              );
	add("recv",               TFA(l_mpi_recv,0)              );
	add("barrier",            TFA(l_mpi_barrier,0)           );
	add("gather",             TFA(l_mpi_gather,0)            );
	add("bcast",              TFA(l_mpi_bcast,0)             );
	add("cart_create",        TFA(l_cart_create,0)           );
	add("comm_split",         TFA(l_mpi_comm_split,0)        );
// 	add("new_request",        l_mpi_newrequest        );
// 	add("isend",              l_mpi_isend             );
// 	add("irecv",              l_mpi_irecv             );
	add("help",               l_mpi_help              );
	add("metatable",          l_mpi_metatable         );
// 	add("next_rank",          l_mpi_next_rank         );
// 	add("prev_rank",          l_mpi_prev_rank         );
// 	add("all2all",            l_mpi_all2all           );

	lua_setglobal(L, "mpi");
}

#endif


#ifdef WIN32
 #ifdef MPI_EXPORTS
  #define MPI_API __declspec(dllexport)
 #else
  #define MPI_API __declspec(dllimport)
 #endif
#else
 #define MPI_API 
#endif


extern "C" {
        #include <lua.h>
        #include <lualib.h>
        #include <lauxlib.h>
}


extern "C"
{
MPI_API int lib_register(lua_State* L);
MPI_API int lib_version(lua_State* L);
MPI_API const char* lib_name(lua_State* L);
MPI_API int lib_main(lua_State* L);	
}


MPI_API int lib_register(lua_State* L)
{
#ifdef _MPI
	registerMPI(L);
#endif
	return 0;
}

MPI_API int lib_version(lua_State* L)
{
	return __revi;
}


MPI_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "MPI";
#else
	return "MPI-Debug";
#endif
}

MPI_API int lib_main(lua_State* L)
{
	return 0;
}


