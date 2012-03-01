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

template<int base>
static int l_mpi_get_rank(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	lua_pushinteger(L, mpi_rank(comm)+1);
	return 1;
}


#if 0
template<int base>
static int l_mpi_next_rank(lua_State* L)
{
	int r;
	if(lua_isnumber(L, 1))
		r = lua_tointeger(L, 1);
	else
		r = mpi_rank();

	const int s = mpi_size(comm);
		
	lua_pushinteger(L, ((r+1)%s)+1);
	return 1;
}

template<int base>
static int l_mpi_prev_rank(lua_State* L)
{
	int r;
	if(lua_isnumber(L, 1))
		r = lua_tointeger(L, 1);
	else
		r = mpi_rank();

	const int s = mpi_size(comm);
		
	lua_pushinteger(L, ((r-1+s)%s)+1);
	return 1;
}
#endif
	
template<int base>
static int l_mpi_barrier(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	MPI_Barrier(comm);
	return 0;
}

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

#if 0
static int l_mpi_gather(lua_State* L)
{
	int root = lua_tointeger(L, 1) - 1; //lua is base 1
	int n = mpi_size(comm);

	char* buf = 0;
	int bufsize = 0;
	
	int reqBufSize;
	MPI_Status stat;
	
	if(mpi_rank() == root)
	{
		lua_newtable(L); //the return table
		for(int i=0; i<n; i++)
		{
			if(i == root)
			{
				if(!lua_istable(L, -2))
					return luaL_error(L, "Gather data from %d was not a table", i+1);
				lua_pushvalue(L, -2); //push a copy of the input
			}
			else
			{
				MPI_Recv(&reqBufSize, 1, MPI_INT, i, BUFSIZE_TAG+i+GATHER_STEP, MPI_COMM_WORLD, &stat);
				if(reqBufSize > bufsize)
				{
					buf = (char*)realloc(buf, reqBufSize);
					bufsize = reqBufSize;
				}
		
				MPI_Recv(buf, reqBufSize, MPI_CHAR, i, LUAVAR_TAG+i+GATHER_STEP, MPI_COMM_WORLD, &stat);
				importLuaVariable(L, buf, reqBufSize);
				if(!lua_istable(L, -1))
					return luaL_error(L, "Gather data from %d was not a table", i+1);
			}

			//add table values to return table
			lua_pushnil(L);
			while(lua_next(L, -2) != 0)
			{
				lua_pushvalue(L, -2); //copy key and value
				lua_pushvalue(L, -2);

				lua_settable(L, -6); //reach past copy, orig, sourcetable

				lua_pop(L, 1); //pop original value
			}
			lua_pop( L, 1 ); //pop nil
		}
	}
	else
	{
		if(!lua_istable(L, -1))
			return luaL_error(L, "sent item in gather must be a table");
					
		int size;	
		char* buf = exportLuaVariable(L, -1, &size);
		
		MPI_Send(&size, 1, MPI_INT, root, BUFSIZE_TAG+mpi_rank()+GATHER_STEP, MPI_COMM_WORLD);
		MPI_Send(buf, size, MPI_CHAR, root, LUAVAR_TAG+mpi_rank()+GATHER_STEP, MPI_COMM_WORLD);
		free(buf);

		//returning local slice
	}

	return 1;
}
#endif

#if 0
typedef struct mpi_req //non-blocking io
{
	char* data;
	MPI_Request request;
	int freedata;
	int refcount;
	int inuse;
} mpi_req;

mpi_req* checkRequest(lua_State* L, int idx)
{
    mpi_req** p = (mpi_req**)luaL_checkudata(L, idx, "MPI.request");
    luaL_argcheck(L, p != NULL, 1, "`MPI_Request' expected");
    return *p;
}

int l_mpi_newrequest(lua_State* L)
{
	mpi_req* io = new mpi_req;
	io->refcount = 1;
	io->data = 0;
	io->freedata = 0;
	io->inuse = 0;

	mpi_req** pp = (mpi_req**)lua_newuserdata(L, sizeof(mpi_req**));

    *pp = io;
    luaL_getmetatable(L, "MPI.request");
    lua_setmetatable(L, -2);
    return 1;
}


int l_mpirequest_tostring(lua_State* L)
{
	mpi_req* io = checkRequest(L, 1);
	if(!io) return 0;

	/*
	if(io->inuse)
	{
		MPI_Status status;
		int flag;
		MPI_Test(&(io->request), &flag, &status);
		lua_pushboolean(L, flag);
	}
	*/
	lua_pushstring(L, "MPI_Request");
	return 1;
}

int l_mpi_isend(lua_State* L)
{
	int dest = lua_tointeger(L, 1) - 1; //lua is base 1

	mpi_req* io = checkRequest(L, -1);

	if(!io)
		return luaL_error(L, "Last argument of isend must be an MPI_Request");

	if(dest < 0 || dest >= mpi_size(comm))
		return luaL_error(L, "Send destination (%d) is out of range.", dest+1); 
	
	int n = lua_gettop(L) - 1;
	int size;
	char* buf;
	
	MPI_Send(&n, 1, MPI_INT, dest, NUMLUAVAR_TAG, MPI_COMM_WORLD);
	
	for(int i=0; i<n; i++)
	{
		buf = exportLuaVariable(L, i+2, &size);
		
		MPI_Send(&size, 1, MPI_INT, dest, BUFSIZE_TAG+i, MPI_COMM_WORLD);
		MPI_Send(buf, size, MPI_CHAR, dest, LUAVAR_TAG+i, MPI_COMM_WORLD);
		free(buf);
	}
	
}

int l_mpi_irecv(lua_State* L)
{
	return 0;
}
#endif

#if 0
static int pos_val(lua_State* L, int neg)
{
	// -1 = n
	// -2 = n-1
	// -3 = n-2
	return lua_gettop(L) + neg + 1;
}

static int merge_data(lua_State* L, int dest, int src)
{
	lua_pushnil(L);

	while(lua_next(L, src))
	{
		if(lua_istable(L, -1))
		{
			lua_pushvalue(L, -2);
			lua_gettable(L, dest); //get value at dest[key]

			if(lua_isnil(L, -1))
			{
				lua_pop(L, 1);
				lua_newtable(L);
			}

			merge_data(L, pos_val(L, -1), pos_val(L, -2));
			lua_pushvalue(L, -3); //repush key
			lua_insert(L, -2); //push key under dest_val
			lua_settable(L, dest);
		}
		else
		{
			lua_pushvalue(L, -2); //copy key, value
			lua_pushvalue(L, -2);
			lua_settable(L, dest);
		}
		lua_pop(L, 1);
	}
}
#endif

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

	importLuaVariable(L, buf, size);

	free(buf);
	return 1;
}



#if 0
int l_mpi_all2all(lua_State* L)
{
	int s = mpi_size(comm);
	int n = lua_gettop(L);

	char* buf;
	int size;

	int* remote_sizes = new int [s];
	buf = exportLuaVariable(L, 1, &size);

	lua_pop(L, n); //clear stack

	MPI_Alltoall(&size, 1, MPI_INT, remote_sizes, 1, MPI_INT, comm);

	int max_chunk_size = 0;
	for(int i=0; i<s; i++)
	{
		if(remote_sizes[i] > max_chunk_size)
			max_chunk_size = remote_sizes[i];
	}

	char*  b = new char[max_chunk_size];
	char* tb = new char[max_chunk_size * s];

	memcpy(b, buf, size);

	MPI_Alltoall(buf, max_chunk_size, MPI_CHAR, tb, 1, MPI_CHAR, comm);

	// now we have all the individual data in remote_bufs[], need to union them
	// lets start with an empty table

	lua_newtable(L);
	for(int i=0; i<s; i++)
	{
		importLuaVariable(L, tb + i*s, remote_sizes[i]);
		
		merge_data(L, 1, 2);
		lua_pop(L, 1);
	}

	free(buf);
	delete [] b;
	delete [] tb;
	delete [] remote_sizes;

	return 1;
}
#endif

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
	if(func == &(l_mpi_get_size<0>) || func == &(l_mpi_get_size<0>))
	{
		lua_pushstring(L, "Return the total number of proccesses in the global workgroup");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 Number: Number of processes");
		return 3;
	}	
	if(func == &(l_mpi_get_rank<0>) || func == &(l_mpi_get_rank<1>))
	{
		lua_pushstring(L, "The rank of the calling process");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 Number: Rank");
		return 3;
	}	
	if(func == &(l_mpi_send<0>) || func == &(l_mpi_send<1>))
	{
		lua_pushstring(L, "Send data to another process in the workgroup");
		lua_pushstring(L, "1 Number, ...: Index of remote process followed by zero or more variables"); 
		lua_pushstring(L, "");
		return 3;
	}	
	if(func == &(l_mpi_recv<0>) || func == &(l_mpi_recv<1>))
	{
		lua_pushstring(L, "Receive data from another process in the workgroup");
		lua_pushstring(L, "1 Number: Index of remote process sending the data"); 
		lua_pushstring(L, "...: zero or more pieces of data");
		return 3;
	}	
	if(func == &(l_mpi_barrier<0>) || func == &(l_mpi_barrier<1>))
	{
		lua_pushstring(L, "Syncronization barrier");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "");
		return 3;
	}
	
#if 0
	if(func == l_mpi_next_rank)
	{
		lua_pushstring(L, "Return the rank of the next process with periodic bounds.");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 Number: Next rank");
		return 3;
	}
	if(func == l_mpi_prev_rank)
	{
		lua_pushstring(L, "Return the rank of the previous process with periodic bounds.");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 Number: Previous rank");
		return 3;
	}
#endif
#if 0

	if(func == l_mpi_all2all)
	{
		lua_pushstring(L, "Return the rank of the previous process with periodic bounds.");
		lua_pushstring(L, "1 Table: Local table to share"); 
		lua_pushstring(L, "1 Table: Union of all other tables");
		return 3;
	}
#endif

	if(func == &(l_mpi_gather<0>) || func == &(l_mpi_gather<1>))
	{
		lua_pushstring(L, "Gather data to a given rank. The return at the rank will be a table with each data as the value for source keys.");
		lua_pushstring(L, "1 value, 1 integer: The value is what will be gathered, the integer is the rank where the data will be gathered."); 
		lua_pushstring(L, "1 table or nil. The table will be returned at the given rank otherwise nil.");
		return 3;
	}
	
	if(func == &(l_mpi_bcast<0>) || func == &(l_mpi_bcast<1>))
	{
		lua_pushstring(L, "Broadcast data to all members of the workgrroup from the given rank.");
		lua_pushstring(L, "1 value, 1 integer: The value is what will be broadcasted, the integer is the rank where the data will be from."); 
		lua_pushstring(L, "1 value. The broadcasted data.");
		return 3;
	}
	
	if(func == &(l_cart_create<0>) || func == &(l_cart_create<1>))
	{
		lua_pushstring(L, "Create a cartesian workgroup.");
		lua_pushstring(L, "2 Tables, 1 Boolean: The first table contains sizes of each dimension, the second is a table of booleans indicating if each dimension should be periodic. The last boolean argument states if the ranks may be reordered"); 
		lua_pushstring(L, "1 MPI_Comm: Optimized for cartesian communication");
		return 3;
	}
#if 0
	if(func == l_mpi_newrequest)
	{
		lua_pushstring(L, "");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_mpi_isend)
	{
		lua_pushstring(L, "");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_mpi_irecv)
	{
		lua_pushstring(L, "");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "");
		return 3;
	}
#endif
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
	add("get_size",           l_mpi_get_size<1>   );
	add("get_rank",           l_mpi_get_rank<1>   );
	add("send",               l_mpi_send<1>       );
	add("recv",               l_mpi_recv<1>       );
	add("barrier",            l_mpi_barrier<1>    );
	add("gather",             l_mpi_gather<1>     );
	add("bcast",              l_mpi_bcast<1>      );
	add("__gc",               l_MPI_Comm_gc       );
	add("__tostring",         l_MPI_Comm_tostring );
	add("cart_create",        l_cart_create<1>    );
	lua_pop(L,1); //metatable is registered
	

	lua_newtable(L);
	add("get_processor_name", l_mpi_get_processor_name);
	add("get_size",           l_mpi_get_size<0>          );
	add("get_rank",           l_mpi_get_rank<0>          );
	add("send",               l_mpi_send<0>              );
	add("recv",               l_mpi_recv<0>              );
	add("barrier",            l_mpi_barrier<0>           );
	add("gather",             l_mpi_gather<0>            );
	add("bcast",              l_mpi_bcast<0>             );
	add("cart_create",        l_cart_create<0>           );
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


