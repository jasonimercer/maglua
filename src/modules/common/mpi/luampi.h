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


#ifndef MPICODE_DEF
#define MPICODE_DEF

#ifdef WIN32
 #ifdef MPI_EXPORTS
  #define MPI_API __declspec(dllexport)
 #else
  #define MPI_API __declspec(dllimport)
 #endif
#else
 #define MPI_API 
#endif



void registerMPI(lua_State* L);

#include "luabaseobject.h"
class MPI_API lua_mpi_request : public LuaBaseObject
{
public:
	lua_mpi_request();
	~lua_mpi_request();
	
	LINEAGE1("mpi.request")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);

	void allocateRecvBuffer();
	int test();
	int ltest(lua_State* L);
	int data(lua_State* L);
	void cancel();
	void wait();

	void recv_init();

	bool sender; //flag for participant role
	
    MPI_Request request[4]; // 1 10 100 N
    int active_requests;
	int active[4];
	char* buf;
	char* full_buf;
	int buf_size;
	int tag;
	int source, dest;
	MPI_Comm comm;

	bool got_first_chunk;
};

#endif
#endif
