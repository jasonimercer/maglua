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

extern "C" {
        #include <lua.h>
        #include <lualib.h>
        #include <lauxlib.h>
}

#ifdef _MPI
#include <mpi.h>
#include <stdlib.h>
#include "luampi.h"
#include "luamigrate.h"

#define NUMLUAVAR_TAG 100
#define    LUAVAR_TAG 200
#define   BUFSIZE_TAG 300
#define   GATHER_STEP 1000

inline int mpi_rank()
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	return rank;
}



inline int mpi_size()
{
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	return size;
}

static int l_mpi_get_processor_name(lua_State* L)
{
	int  namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	
	MPI_Get_processor_name(processor_name, &namelen);
	
	lua_pushstring(L, processor_name);
	return 1;
}

static int l_mpi_get_size(lua_State* L)
{
	lua_pushinteger(L, mpi_size());
	return 1;
}

static int l_mpi_get_rank(lua_State* L)
{
	lua_pushinteger(L, mpi_rank()+1);
	return 1;
}

static int l_mpi_next_rank(lua_State* L)
{
	int r;
	if(lua_isnumber(L, 1))
		r = lua_tonumber(L, 1);
	else
		r = mpi_rank();

	const int s = mpi_size();
		
	lua_pushinteger(L, ((r+1)%s)+1);
	return 1;
}

static int l_mpi_prev_rank(lua_State* L)
{
	int r;
	if(lua_isnumber(L, 1))
		r = lua_tonumber(L, 1);
	else
		r = mpi_rank();

	const int s = mpi_size();
		
	lua_pushinteger(L, ((r-1+s)%s)+1);
	return 1;
}
	
static int l_mpi_barrier(lua_State* L)
{
	MPI_Barrier(MPI_COMM_WORLD);
	return 0;
}

static int l_mpi_send(lua_State* L)
{
	int dest = lua_tointeger(L, 1) - 1; //lua is base 1

	
	if(dest < 0 || dest >= mpi_size())
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
	return 0;
}

static int l_mpi_recv(lua_State* L)
{
	int src = lua_tointeger(L, 1) - 1; //lua is base 1
	int n;
	MPI_Status stat;

	if(src < 0 || src >= mpi_size())
		return luaL_error(L, "Receive source (%d) is out of range.", src+1);

	char* buf = 0;
	int bufsize = 0;
	
	int reqBufSize;
	
	MPI_Recv(&n, 1, MPI_INT, src, NUMLUAVAR_TAG, MPI_COMM_WORLD, &stat);
	for(int i=0; i<n; i++)
	{
		MPI_Recv(&reqBufSize, 1, MPI_INT, src, BUFSIZE_TAG+i, MPI_COMM_WORLD, &stat);
		if(reqBufSize > bufsize)
		{
			buf = (char*)realloc(buf, reqBufSize);
			bufsize = reqBufSize;
		}
		
		MPI_Recv(buf, reqBufSize, MPI_CHAR, src, LUAVAR_TAG+i, MPI_COMM_WORLD, &stat);
		
		importLuaVariable(L, buf, reqBufSize);
	}
	
	if(buf)
		free(buf);
	return n;
}


static int l_mpi_gather(lua_State* L)
{
	int root = lua_tointeger(L, 1) - 1; //lua is base 1
	int n = mpi_size();

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

	if(dest < 0 || dest >= mpi_size())
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
	if(func == l_mpi_get_size)
	{
		lua_pushstring(L, "Return the total number of proccesses in the global workgroup");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 Number: Number of processes");
		return 3;
	}	
	if(func == l_mpi_get_rank)
	{
		lua_pushstring(L, "The rank of the calling process");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 Number: Rank");
		return 3;
	}	
	if(func == l_mpi_send)
	{
		lua_pushstring(L, "Send data to another process in the workgroup");
		lua_pushstring(L, "1 Number, ...: Index of remote process followed by zero or more variables"); 
		lua_pushstring(L, "");
		return 3;
	}	
	if(func == l_mpi_recv)
	{
		lua_pushstring(L, "Receive data from another process in the workgroup");
		lua_pushstring(L, "1 Number: Index of remote process sending the data"); 
		lua_pushstring(L, "...: zero or more pieces of data");
		return 3;
	}	
	if(func == l_mpi_barrier)
	{
		lua_pushstring(L, "Syncronization barrier");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "");
		return 3;
	}
	
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
	
#if 0
	if(func == l_mpi_gather)
	{
		lua_pushstring(L, "");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "");
		return 3;
	}
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




int l_mpirequest_gc(lua_State* L)
{
	mpi_req* io = checkRequest(L, 1);
	if(!io) return 0;

	io->refcount--;
	if(io->refcount == 0)
		delete io;

	return 0;
}

#define add(name, func) \
	lua_pushstring(L, name); \
	lua_pushcfunction(L, func); \
	lua_settable(L, -3); 

void registerMPI(lua_State* L)
{
	static const struct luaL_reg mpirequest_m [] = { //methods
		{"__gc",         l_mpirequest_gc},
		{"__tostring",   l_mpirequest_tostring},
		{NULL, NULL}
	};

    luaL_newmetatable(L, "MPI.request");
    lua_pushstring(L, "__index");
    lua_pushvalue(L, -2);  /* pushes the metatable */
    lua_settable(L, -3);  /* metatable.__index = metatable */
    luaL_register(L, NULL, mpirequest_m);
    lua_pop(L,1); //metatable is registered


	lua_newtable(L);
	add("get_processor_name", l_mpi_get_processor_name);
	add("get_size",           l_mpi_get_size          );
	add("get_rank",           l_mpi_get_rank          );
	add("send",               l_mpi_send              );
	add("recv",               l_mpi_recv              );
	add("barrier",            l_mpi_barrier           );
	add("gather",             l_mpi_gather            );
	add("new_request",        l_mpi_newrequest        );
	add("isend",              l_mpi_isend             );
	add("irecv",              l_mpi_irecv             );
	add("help",               l_mpi_help              );
	add("metatable",          l_mpi_metatable         );
	add("next_rank",          l_mpi_next_rank         );
	add("prev_rank",          l_mpi_prev_rank         );

	lua_setglobal(L, "mpi");
}

#endif


extern "C"
{
int lib_register(lua_State* L)
{
#ifdef _MPI
	registerMPI(L);
#endif
	return 0;
}

int lib_deps(lua_State* L)
{
	return 0;
}
}
