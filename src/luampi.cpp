#ifdef _MPI
#include <mpi.h>
#include <stdlib.h>
#include "luampi.h"
#include "luamigrate.h"

#define NUMLUAVAR_TAG 1
#define    LUAVAR_TAG 2
#define   BUFSIZE_TAG 3

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
	int numprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	
	lua_pushinteger(L, numprocs);
	return 1;
}

static int l_mpi_get_rank(lua_State* L)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	lua_pushinteger(L, rank+1);
	return 1;
}

static int l_mpi_test(lua_State* L)
{
	char* buf;
	int size;
	
	buf = exportLuaVariable(L, 1, &size);

	importLuaVariable(L, buf, size);

	free(buf);
	return 1;
}

static int l_mpi_send(lua_State* L)
{
	int dest = lua_tointeger(L, 1) - 1; //lua is base 1
	int n = lua_gettop(L) - 1;
	int size;
	char* buf;
	
	MPI_Send(&n, 1, MPI_INT, dest, NUMLUAVAR_TAG, MPI_COMM_WORLD); 
	
	for(int i=0; i<n; i++)
	{
		buf = exportLuaVariable(L, i+2, &size);
		MPI_Send(&size, 1, MPI_INT, dest, BUFSIZE_TAG, MPI_COMM_WORLD);
		MPI_Send(buf, size, MPI_CHAR, dest, LUAVAR_TAG, MPI_COMM_WORLD);
		free(buf);
	}
	return 0;
}

static int l_mpi_recv(lua_State* L)
{
	int src = lua_tointeger(L, 1) - 1; //lua is base 1
	int n;
	MPI_Status stat;

	char* buf = 0;
	int bufsize = 0;
	
	int reqBufSize;
	
	int r = MPI_Recv(&n, 1, MPI_INT, src, NUMLUAVAR_TAG, MPI_COMM_WORLD, &stat);
	
	for(int i=0; i<n; i++)
	{
		MPI_Recv(&reqBufSize, 1, MPI_INT, src, BUFSIZE_TAG, MPI_COMM_WORLD, &stat);
		if(reqBufSize > bufsize)
		{
			buf = (char*)realloc(buf, reqBufSize);
			bufsize = reqBufSize;
		}
		
		MPI_Recv(buf, reqBufSize, MPI_CHAR, src, LUAVAR_TAG, MPI_COMM_WORLD, &stat);
		
		importLuaVariable(L, buf, reqBufSize);
	}
	
	if(buf)
		free(buf);
	return n;
}


#define add(name, func) \
	lua_pushstring(L, name); \
	lua_pushcfunction(L, func); \
	lua_settable(L, -3); 

void registerMPI(lua_State* L)
{
	lua_newtable(L);
	
	add("get_processor_name", l_mpi_get_processor_name);
	add("get_size",           l_mpi_get_size          );
	add("get_rank",           l_mpi_get_rank          );
	add("send",               l_mpi_send              );
	add("recv",               l_mpi_recv              );
	add("test",               l_mpi_test              );
	
	lua_setglobal(L, "mpi");
}

#endif
