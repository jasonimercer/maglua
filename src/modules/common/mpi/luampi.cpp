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
// GCC < 4.5 has buggy template pointers

#define CHUNK_SIZE 1024
#define FIRSTCHUNK_TAG 98
#define REMAININGCHUNK_TAG 99
#define NUMLUAVAR_TAG 100
#define    LUAVAR_TAG 200
#define   BUFSIZE_TAG 300
#define   GATHER_STEP 1000

#define REQ1_SIZE (CHUNK_SIZE    )
#define REQ2_SIZE (CHUNK_SIZE*10 )
#define REQ3_SIZE (CHUNK_SIZE*100)

#define REQ12_SIZE (REQ1_SIZE + REQ2_SIZE)
#define REQ123_SIZE (REQ12_SIZE + REQ3_SIZE)

extern "C" {
        #include <lua.h>
        #include <lualib.h>
        #include <lauxlib.h>
}


static int getSendBufferSize(const char* buf);
static void setSendBufferSize(char* buf, int sz);
static char* setupSendBuffer(lua_State* L, int export_start_idx, int export_end_idx, int min_buf_size=-1);


static int decodeBuffer(lua_State* L, char* buf);

static char* realloc_to_req_size(char* src, int& size)
{
	if(size > REQ12_SIZE && size < REQ123_SIZE)
	{
		size = REQ123_SIZE;
		return (char*)realloc( (void*)src, REQ123_SIZE);
	}
	if(size > REQ1_SIZE && size < REQ12_SIZE)
	{
		size = REQ12_SIZE;
		return (char*)realloc( (void*)src, REQ12_SIZE);
	}
	if(size < REQ1_SIZE)
	{
		size = REQ1_SIZE;
		return (char*)realloc( (void*)src, REQ1_SIZE);
	}

	// else it's big. leave it big
	return src;
}


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
	mpi_comm_lua(MPI_Comm c) {comm = c; refcount=0;}
	MPI_Comm comm;
    int refcount;
} mpi_comm_lua;

mpi_comm_lua* checkMPI_Comm(lua_State* L, int idx)
{
    mpi_comm_lua** pp = (mpi_comm_lua**)luaL_checkudata(L, idx, "mpi");
    luaL_argcheck(L, pp != NULL, 1, "`MPI_Comm' expected");
    return *pp;
}

void lua_pushMPI_Comm(lua_State* L, mpi_comm_lua* p)
{
    p->refcount++;
    mpi_comm_lua** pp = (mpi_comm_lua**)lua_newuserdata(L, sizeof(mpi_comm_lua**));
    *pp = p;
    luaL_getmetatable(L, "mpi");
    lua_setmetatable(L, -2);
}
void lua_pushMPI_Comm(lua_State* L, MPI_Comm c)
{
	if(c == MPI_COMM_NULL)
	{
		lua_pushnil(L);
	}
	else
	{
		lua_pushMPI_Comm(L, new mpi_comm_lua(c));
	}
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


lua_mpi_request::lua_mpi_request()
	: LuaBaseObject(hash32("lua_mpi_request"))
{
	buf = 0;
	full_buf = 0;
	buf_size = 0;
	active_requests = 0;
	got_first_chunk = false;

	active[0] = 0;
	active[1] = 0;
	active[2] = 0;
	active[3] = 0;

	sender = false;
}

lua_mpi_request::~lua_mpi_request()
{
	cancel();
	if(buf)
		free(buf);
	if(full_buf)
		free(full_buf);	
}

int lua_mpi_request::luaInit(lua_State* L)
{
	return LuaBaseObject::luaInit(L);
}

void lua_mpi_request::allocateRecvBuffer()
{
	buf = (char*)malloc(REQ123_SIZE);
	full_buf = 0; //for 4th chunk
}

void lua_mpi_request::recv_init()
{
	if(got_first_chunk)
		return;
	if(sender)
		return;

	int flag;
	MPI_Status status;

	MPI_Test(&(request[0]), &flag, &status);

	if(flag) // then we have the 1st chunk and we can calculate how many chunks are coming in and how large #4 might be
	{
		// 1st chunk in buf
		active[0] = 0;
		const int full_buffer_size = *((int*)buf);
		active_requests = 1; //this is a given

		// printf("receiving %i\n", full_buffer_size);

		if(full_buffer_size > REQ1_SIZE)
			active_requests++;
		else
		{
			MPI_Cancel(&(request[1])); //don't need this active request
			active[1] = 0;
		}
		if(full_buffer_size > REQ12_SIZE)
			active_requests++;
		else
		{
			MPI_Cancel(&(request[2])); //don't need this active request
			active[2] = 0;
		}
		if(full_buffer_size > REQ123_SIZE)
		{
			active_requests++;
			full_buf = (char*)malloc(full_buffer_size);
			MPI_Irecv(full_buf + REQ123_SIZE, full_buffer_size - REQ123_SIZE, MPI_CHAR, source, tag+3, comm, &(request[3]));
			active[3] = 1;
		}
		// no else statement to cancel because 4th chunk is only started if needed

		got_first_chunk = true;
	}

}


int lua_mpi_request::test()
{
	int flag;
	MPI_Status status;

	if(sender)
	{
		for(int i=0; i<active_requests; i++)
		{
			MPI_Test(&(request[i]), &flag, &status);
			if(!flag)
			{
				return 0;
			}
		}
		return 1;
	}
	//else
	
	recv_init();
	if(!got_first_chunk)
	{
		return 0;
	}
	
	for(int i=0; i<active_requests; i++)
	{
		if(active[i])
		{
			MPI_Test(&(request[i]), &flag, &status);
			if(!flag)
				return 0;
		}
		active[i] = 0;
	}
	return 1;
}

int lua_mpi_request::ltest(lua_State* L)
{
	lua_pushboolean(L, test());
	return 1;
}

void lua_mpi_request::wait()
{
	MPI_Status status;
	if(sender)
	{
		for(int i=0; i<4; i++)
		{
			if(active[i])
			{
				MPI_Wait(&(request[i]), &status);
			}
			active[i] = 0;
		}
		return;
	}
	//else
	
	// test();
	for(int i=0; i<4; i++)
	{
		recv_init();
		if(active[i])
		{
			MPI_Wait(&(request[i]), &status);
		}
		active[i] = 0;
	}
}

void lua_mpi_request::cancel()
{
	for(int i=0; i<4; i++)
	{
		if(active[i])
		{
			MPI_Cancel(&(request[i]));
		}
		active[i] = 0;
	}
}

int lua_mpi_request::data(lua_State* L)
{
	if(sender)
		return decodeBuffer(L, buf);

	if(test()) // then we've recv_init'd and all available chunks are received
	{
		if(full_buf) // then we've got lots 'o data and need to merge the lower 3 with the 4th
		{
			if(buf)
			{
				memcpy(full_buf, buf, REQ123_SIZE);
				free(buf);
				buf = 0;
			}
			return decodeBuffer(L, full_buf);
		}
		else
		{
			return decodeBuffer(L, buf);
		}
	}
	
	return 0;
}

	
	
static int l_rwait(lua_State* L)
{
	LUA_PREAMBLE(lua_mpi_request, r, 1);
	r->wait();
	return 0;
}
static int l_rcancel(lua_State* L)
{
	LUA_PREAMBLE(lua_mpi_request, r, 1);
	r->cancel();
	return 0;
}

static int l_rtest(lua_State* L)
{
	LUA_PREAMBLE(lua_mpi_request, r, 1);
	return r->ltest(L);
}
static int l_rdata(lua_State* L)
{
	LUA_PREAMBLE(lua_mpi_request, r, 1);
	return r->data(L);
}

int lua_mpi_request::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "A request object represents a pending asynchronous communication event");
		lua_pushstring(L, "");
		lua_pushstring(L, ""); //output, empty
		return 3;
	}

	lua_CFunction func = lua_tocfunction(L, 1);

	if(func == l_rwait)
	{
		lua_pushstring(L, "Wait for pending communication to complete. This will block.");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_rtest)
	{
		lua_pushstring(L, "Test to see if a pending communication is complete, this will not block.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 boolean: true if complete, false otherwise.");
		return 3;
	}

	if(func == l_rcancel)
	{
		lua_pushstring(L, "Cancel pending communication");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == l_rdata)
	{
		lua_pushstring(L, "Retrieve the data if available");
		lua_pushstring(L, "");
		lua_pushstring(L, "...: Sent data if the communication is complete, nil otherwise. This will return a copy of the sent data on the sending side.");
		return 3;
	}

	return LuaBaseObject::help(L);


}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* lua_mpi_request::luaMethods()
{
	if(m[127].name)return m;

	static const luaL_Reg _m[] =
	{
		{"wait",      l_rwait},
		{"cancel",    l_rcancel},
		{"test",      l_rtest},
		{"data",      l_rdata},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}

	
#if 0
	LINEAGE1("mpi.request")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
#endif


static int decodeBuffer(lua_State* L, char* buf)
{
	if(buf == 0)
		return 0;

	buffer b;

	b.buf = buf + sizeof(int)*2;
	b.size = getSendBufferSize(buf) - sizeof(int)*2;
	b.pos = 0;

	int n;
	memcpy(&n, buf+sizeof(int), sizeof(int));

	for(int i=0; i<n; i++)
	{
		_importLuaVariable(L, &b);
	}

	buffer_unref(L, &b);

	return n;
}

static int getSendBufferSize(const char* buf)
{
	int i;
	memcpy(&i, buf, sizeof(int));
	return i;
}

static void setSendBufferSize(char* buf, int sz)
{
	memcpy(buf, &sz, sizeof(int));
}

// inclusive endpoints
static char* setupSendBuffer(lua_State* L, int export_start_idx, int export_end_idx, int min_buf_size)
{
	int n = export_end_idx - export_start_idx + 1;
	// printf("SEND BUFFER SIZE: %i\n", n);
	buffer b;
	buffer_init(&b);

	for(int i=0; i<n; i++)
	{
		int j = i+export_start_idx;
		if(lua_isnone(L, j))
		{
			lua_pushnil(L);
			_exportLuaVariable(L, lua_gettop(L), &b);
			lua_pop(L, 1);
		}
		else
		{
			_exportLuaVariable(L, j, &b);
		}
	}


	int full_buffer_size = b.pos + sizeof(int)*2;

	// we may want the full buffer size to fit in some ranges
	if(min_buf_size > 0)
	{
		if(full_buffer_size < min_buf_size)
			full_buffer_size = min_buf_size;
	}

	char* full_buffer = (char*)malloc(full_buffer_size);

	int pos = 0;
	memcpy(full_buffer + pos, &full_buffer_size, sizeof(int)); pos+=sizeof(int);
	memcpy(full_buffer + pos, &n,                sizeof(int)); pos+=sizeof(int);
	memcpy(full_buffer + pos, b.buf, b.pos);
	
	buffer_unref(L, &b);
	free(b.buf);
	
	return full_buffer;
}

#define IBUFCHUNK (sizeof(int)*16)

template<int base>
static int l_mpi_isend(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	int dest = lua_tointeger(L, base+1) - 1; //lua is base 1
	
	int tag = lua_tointeger(L, base+2) * 4;

	if(dest < 0 || dest >= mpi_size(comm))
		return luaL_error(L, "ISend destination (%d) is out of range.", dest+1); 
	
	// int n;
	// int complete_message_size;
	// int error;
	// +1 = destination
	// +2 = tag
	// +3 to end = data
	char* full_buffer = setupSendBuffer(L, base+3, lua_gettop(L), IBUFCHUNK);

	int complete_message_size = getSendBufferSize(full_buffer);

	full_buffer = realloc_to_req_size(full_buffer, complete_message_size);
	
	setSendBufferSize(full_buffer, complete_message_size);

	lua_mpi_request* req = new lua_mpi_request;
	
	req->buf = full_buffer;
	req->buf_size = complete_message_size;
	req->active_requests = 0;
	req->tag = tag;
	req->dest = dest; 
	req->sender = true;

	MPI_Isend(full_buffer + 0, REQ1_SIZE, MPI_CHAR, dest, tag+0, comm, &(req->request[0]));
	req->active[0] = 1;

	if(complete_message_size > REQ1_SIZE)
	{
		MPI_Isend(full_buffer + REQ1_SIZE, REQ2_SIZE, MPI_CHAR, dest, tag+1, comm, &(req->request[1]));
		req->active[1] = 1;
	}
	if(complete_message_size > REQ12_SIZE)
	{
		MPI_Isend(full_buffer + REQ12_SIZE, REQ3_SIZE, MPI_CHAR, dest, tag+1, comm, &(req->request[2]));
		req->active[2] = 1;
	}
	if(complete_message_size > REQ123_SIZE)
	{
		MPI_Isend(full_buffer + REQ123_SIZE, complete_message_size - REQ12_SIZE, MPI_CHAR, dest, tag+1, comm, &(req->request[2]));
		req->active[3] = 1;
	}

	luaT_push<lua_mpi_request>(L, req);
	
	return 1;
}


template<int base>
static int l_mpi_irecv(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	int source = lua_tointeger(L, base+1) - 1; //lua is base 1
	
	int tag = lua_tointeger(L, base+2) * 4;

	if(source < 0 || source >= mpi_size(comm))
		return luaL_error(L, "IRecv source (%d) is out of range.", source+1); 

	lua_mpi_request* req = new lua_mpi_request;
	
	req->active_requests = 3;
	req->sender = false;
	req->allocateRecvBuffer();
	req->tag = tag;
	req->source = source;
	req->comm = comm;
	char* buf = req->buf;

	int r1 = MPI_Irecv(buf,              REQ1_SIZE, MPI_CHAR, source, tag+0, comm, &(req->request[0]));
	int r2 = MPI_Irecv(buf + REQ1_SIZE,  REQ2_SIZE, MPI_CHAR, source, tag+1, comm, &(req->request[1]));
	int r3 = MPI_Irecv(buf + REQ12_SIZE, REQ3_SIZE, MPI_CHAR, source, tag+2, comm, &(req->request[2]));
	// printf("%i %i %i\n", r1, r2, r3);

	req->active[0] = 1;
	req->active[1] = 1;
	req->active[2] = 1;
	req->active[3] = 0;

	luaT_push<lua_mpi_request>(L, req);
	
	return 1;
}
















template<int base>
static int l_mpi_comm_split(lua_State* L)
{
	MPI_Comm old = get_comm<base>(L);

	int colour = lua_tointeger(L, base+1);
	
	MPI_Comm new_comm;
	MPI_Comm_split(old, colour, 0, &new_comm);
	lua_pushMPI_Comm(L, new_comm);

    return 1;
}

template<int base>
static int l_cart_coord(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);

	int rank = lua_tointeger(L, base+1) - 1;
	int dims = lua_tointeger(L, base+2);

	if(dims <= 0 || dims >= 10)
		return luaL_error(L, "Dimension count must be greater than zero and less than 10");

	int* d = new int[dims];


	if(MPI_Cart_coords(comm, rank, dims, d))
		return luaL_error(L, "Failed to lookup coordinates. Invalid rank or non-cartesian workgroup?");

	lua_newtable(L);
	for(int i=0; i<dims; i++)
	{
		lua_pushinteger(L, i+1);
		lua_pushinteger(L, d[i]+1);
		lua_settable(L, -3);
	}

	delete [] d;
	return 1;
}

template<int base>
static int l_cart_rank(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	
	int coord[10] = {0,0,0,0,0,0,0,0,0,0};
	int rank;
	
	if(!lua_istable(L, base+1))
	{
		return luaL_error(L, "Coordinate must be given in a table");
	}

	for(int i=0; i<10; i++)
	{
		lua_pushinteger(L, i+1);
		lua_gettable(L, base+1);
		if(lua_isnumber(L, -1))
			coord[i] = lua_tointeger(L, -1) - 1;
		lua_pop(L, 1);
	}

	if(MPI_Cart_rank(comm, coord, &rank))
		return luaL_error(L, "Failed to get rank for provided coodinate. Invalid coordinate? Non-cartesian communicator?");

	lua_pushinteger(L, rank+1);
	return 1;

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
	
	int reorder = 1;
	if(!lua_isnone(L, base+3))
		reorder = lua_toboolean(L, base+3);
	
	if(ndims == 0)
		return luaL_error(L, "Zero dimensional grid not allowed");
	
	MPI_Comm new_comm;
	MPI_Cart_create(old, ndims, dims, per, reorder, &new_comm);

	lua_pushMPI_Comm(L, new_comm);

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
	const int r = mpi_rank(comm);
	if(r == MPI_UNDEFINED) // possible when asking about the rank when you're not part of the comm
		return 0;
	lua_pushinteger(L, r+1);
	return 1;
}


template<int base>
static int l_mpi_barrier(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	MPI_Barrier(comm);
	return 0;
}


template<int base>
static int l_mpi_splitrange(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	const double low  = lua_tonumber(L, base+1);
	const double high = lua_tonumber(L, base+2);
	
	double step = 1.0;
	if(lua_isnumber(L, base+3))
	{
		step = lua_tonumber(L, base+3);
	}
	
	lua_getglobal(L, "mpi");
	if(!lua_istable(L, -1))
	{
		return luaL_error(L, "Failed to lookup mpi table");
	}
	
	lua_getfield(L, -1, "_make_range_iterator"); //this should be a function
	if(!lua_isfunction(L, -1))
	{
		return luaL_error(L, "`mpi._make_range_iterator' is not a function");
	}
	
	lua_pushMPI_Comm(L, comm);
	lua_pushnumber(L, low);
	lua_pushnumber(L, high);
	lua_pushnumber(L, step);
	
	lua_call(L, 4, 1);

	return 1;
}


// reworking send
// idea is to send a constant sized large buffer that contains
// the buffer size and the head of the data. If the data is small
// enough, this is all that will need to be sent. Otherwise, send the 
// rest of the data
template<int base>
static int l_mpi_send(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	int dest = lua_tointeger(L, base+1) - 1; //lua is base 1
	
	if(dest < 0 || dest >= mpi_size(comm))
		return luaL_error(L, "Send destination (%d) is out of range.", dest+1); 
	
	int n;
	// +1 is destination
	// +2 to end is data
	char* full_buffer = setupSendBuffer(L, base+2, lua_gettop(L), CHUNK_SIZE);

	int complete_message_size = getSendBufferSize(full_buffer);

	// ok, we've now built the buffer, lets send the first chunk, it may be enough
	MPI_Send(full_buffer, CHUNK_SIZE, MPI_CHAR, dest, FIRSTCHUNK_TAG, comm);

	// finally we will send the rest of the data if there is more to go
	if(complete_message_size >= CHUNK_SIZE) //then there's more
	{
		MPI_Send(full_buffer + CHUNK_SIZE, complete_message_size - CHUNK_SIZE, MPI_CHAR, dest, REMAININGCHUNK_TAG, comm);
	}
	
	free(full_buffer);
	
	return 0;
}

// reworking recv to match new send
template<int base>
static int l_mpi_recv(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	int src = lua_tointeger(L, base+1) - 1; //lua is base 1
	MPI_Status stat;

	if(src < 0 || src >= mpi_size(comm))
		return luaL_error(L, "Receive source (%d) is out of range.", src+1);

	char first_chunk[CHUNK_SIZE];
	MPI_Recv(first_chunk, CHUNK_SIZE, MPI_CHAR, src, FIRSTCHUNK_TAG, comm, &stat);

	int complete_message_size = getSendBufferSize(first_chunk);
	int n;

	if(complete_message_size >= CHUNK_SIZE) //then we need more
	{
		char* buf = new char[complete_message_size];	
		memcpy(buf, first_chunk, CHUNK_SIZE);

	    MPI_Recv(buf + CHUNK_SIZE, complete_message_size - CHUNK_SIZE, MPI_CHAR, src, REMAININGCHUNK_TAG, comm, &stat);

		n = decodeBuffer(L, buf);
		delete [] buf;
	}
	else
	{
		n = decodeBuffer(L, first_chunk);
	}
	
	return n;
}


static void ps(lua_State* L)
{
	for(int i=1; i<=lua_gettop(L); i++)
	{
		printf("%i %s\n", i, lua_typename(L, lua_type(L, i)));
	}
}

template<int base>
int l_mpi_gather(lua_State* L)
{
	MPI_Comm comm = get_comm<base>(L);
	int r = mpi_rank(comm);
	int s = mpi_size(comm);
	int n = lua_gettop(L)-base;

	int root = lua_tointeger(L, base+1)-1;
	//printf("%s:%i\n", __FILE__, __LINE__);
	if(root < 0 || root >= s)
		return luaL_error(L, "invalid rank");

	int* rs = new int [s]; //remote_sizes
	for(int i=0; i<s; i++)
		rs[i] = 0;

	// +1 is root
	char* full_buffer = setupSendBuffer(L, base+2, base+2); // just 1 value

	lua_pop(L, n); //clear stack

	int size = getSendBufferSize(full_buffer);
	int orig_size = size;

	MPI_Gather(&size, sizeof(int), MPI_CHAR, rs, sizeof(int), MPI_CHAR, root, comm);

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

	char* big_chunk = (char*) malloc( ms * s  );
	char* little_chunk = (char*) malloc( ms );

	bzero(big_chunk, ms * s);
	bzero(little_chunk, ms);

	memcpy(little_chunk, full_buffer, orig_size);

	MPI_Gather(little_chunk, ms, MPI_CHAR, big_chunk, ms, MPI_CHAR, root, comm);

	if(r == root)
	{
		lua_newtable(L);
		for(int i=0; i<s; i++)
		{
			char* buf = big_chunk + ms*i;

			lua_pushinteger(L, i+1);
			decodeBuffer(L, buf);
			lua_settable(L, -3);
		}
	}

	free(full_buffer);
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
		// +1 is root
		// +2 to end is data
		buf = setupSendBuffer(L, base+2, lua_gettop(L));
		size = getSendBufferSize(buf);
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
		decodeBuffer(L, buf);
	}
	free(buf);
	return 1;
}


// this nonsense looks dumb but it gets around some compiler problems
static bool FC1(lua_CFunction lhs, lua_CFunction rhs)
{
	return lhs == rhs;
}
// function compare
#define if_FC1(lhs, rhs) if(FC1(lhs,rhs))
#define if_FC2(lhs, rhs1, rhs2) if(FC1(lhs,rhs1) || FC1(lhs,rhs2))


static int l_mpi_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Exposes basic and new convenience MPI functions");
		lua_pushstring(L, ""); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}

	if(!lua_iscfunction(L, 1))
	{
		return 0;
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if_FC1(func, l_mpi_get_processor_name)
	{
		lua_pushstring(L, "Returns the name of the processor as known by MPI.");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 String: Name");
		return 3;
	}
	if_FC2(func, l_mpi_get_size<0>, l_mpi_get_size<1>)
	{
		lua_pushstring(L, "Return the total number of proccesses in the global workgroup");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "1 Number: Number of processes");
		return 3;
	}	
	if_FC2(func, l_mpi_get_rank<0>, l_mpi_get_rank<1>)
	{
		lua_pushstring(L, "The rank of the calling process as a base 1 value. Note: The C and Fortran bindings for MPI use base 0 for ranks, this is automatically translated to base 1 in MagLua for esthetics and language consistency.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: Rank");
		return 3;
	}	
	if_FC2(func, l_mpi_send<0>, l_mpi_send<1>)
	{
		lua_pushstring(L, "Send data to another process in the workgroup. Example:\n<pre>"
"if mpi.get_rank() == 1 then\n"
"	-- sending an anonymous function and some data\n"
"	local msg = \"hello\"\n"
"	mpi.send(2, function(x) print(x,x) end, msg)\n"
"end\n"
"if mpi.get_rank() == 2 then\n"
"	f, x = mpi.recv(1)\n"
"	f(x)\n"
"end\n</pre>"
"Output at process 2:\n"
"<pre>hello	hello</pre>"		
		);
		lua_pushstring(L, "1 Number, ...: Index of remote process followed by zero or more data"); 
		lua_pushstring(L, "");
		return 3;
	}	
	if_FC2(func, l_mpi_recv<0>, l_mpi_recv<1>)
	{
		lua_pushstring(L, "Receive data from another process in the workgroup");
		lua_pushstring(L, "1 Number: Index of remote process sending the data"); 
		lua_pushstring(L, "...: zero or more pieces of data");
		return 3;
	}	
	if_FC2(func, l_mpi_isend<0>, l_mpi_isend<1>)
	{
		lua_pushstring(L, "Asynchronously send data to another process in the workgroups. Example:\n<pre>"
"-- tags ensure that different asynchronous communications don't clash\n"
"local tag = 5\n"
"\n"
"if mpi.get_rank() == 2 then\n"
"    recv_request = mpi.irecv(1, tag)\n"
"end\n"
"\n"
"mpi.barrier() -- barrier to show that an irecv can be before an isend\n"
"\n"
"if mpi.get_rank() == 1 then\n"
"    send_request = mpi.isend(2, tag, \"hello\", 5,6,7)\n"
"end\n"
"\n"
"if mpi.get_rank() == 2 then\n"
"    print(recv_request:data()) -- not guaranteed to print data\n"
"    recv_request:wait()\n"
"    print(recv_request:data()) -- guaranteed to print \"hello   5       6       7\"\n"
"end\n"
"</pre>");
		lua_pushstring(L, "2 Integers, ...: Rank of remote process, tag to be matched on the receiving side followed  by zero or more data"); 
		lua_pushstring(L, "1 *mpi.request*: This request is used to check to see if the communication is complete.");
		return 3;
	}
	if_FC2(func, l_mpi_irecv<0>, l_mpi_irecv<1>)
	{
		lua_pushstring(L, "Asynchronously receive data from another process in the workgroups.");
		lua_pushstring(L, "2 Integers: Rank of remote process, tag to be matched on the sending side.");
		lua_pushstring(L, "1 *mpi.request*: This request is used to check to see if the communication is complete. The data may be retrieved from this object.");
		return 3;
	}
	if_FC2(func, l_mpi_barrier<0>, l_mpi_barrier<1>)
	{
		lua_pushstring(L, "Syncronization barrier");
		lua_pushstring(L, ""); 
		lua_pushstring(L, "");
		return 3;
	}

	if_FC2(func, l_mpi_gather<0>, l_mpi_gather<1>)
	{
		lua_pushstring(L, "Gather data to a given rank. The return at the rank will be a table with each data as the value for source keys.");
		lua_pushstring(L, "1 Integer, 1 value: The value is what will be gathered, the Integer is the rank where the data will be gathered."); 
		lua_pushstring(L, "1 table or nil. The table will be returned at the given rank otherwise nil.");
		return 3;
	}
	
	if_FC2(func, l_mpi_bcast<0>, l_mpi_bcast<1>)
	{
		lua_pushstring(L, "Broadcast data to all members of the workgrroup from the given rank.");
		lua_pushstring(L, "1 Integer, 1 value: The value is what will be broadcasted, the integer is the rank where the data will be from."); 
		lua_pushstring(L, "1 value. The broadcasted data.");
		return 3;
	}
	
	if_FC2(func, l_cart_rank<0>, l_cart_rank<1>)
	{
		lua_pushstring(L, "Lookup rank in a cartesian workgroup based on coordinate.");
		lua_pushstring(L, "1 Table of integers: Coordinate to lookup, values will be wrapped for periodic dimensions.");
		lua_pushstring(L, "1 Integer: Rank of workgroup member at given coodinate.");
		return 3;
	}
	if_FC2(func, l_cart_coord<0>, l_cart_coord<1>)
	{
		lua_pushstring(L, "Lookup coordinate in cartesian workgroup based on rank.");
		lua_pushstring(L, "1 Integer: Rank of workgroup member.");
		lua_pushstring(L, "1 Table of integers: Coordinate at given rank.");
		return 3;
	}



	if_FC2(func, l_cart_create<0>, l_cart_create<1>)
	{
		lua_pushstring(L, "Create a cartesian workgroup, maximum 10 dimensions. 3D Example with periodic bounds in the 1st and 2nd dimension:\n"
		"<pre>mpi_cart = mpi.cart_create({3,3,2}, {true,true,false})\n</pre>");
		lua_pushstring(L, "2 Tables, 1 Optional Boolean: The first table contains sizes of each dimension, the second is a table of booleans indicating if each dimension should be periodic. The last boolean argument states if the ranks may be reordered (default true, some MPI implementations ignore this value)"); 
		lua_pushstring(L, "1 MPI_Comm: Optimized for cartesian communication. In the case that the number of processes in the calling workgroup is larger than the number of processes in the new workgroup, nils will be returned to some members.");
		return 3;
	}
	
	if_FC2(func, l_mpi_comm_split<0>, l_mpi_comm_split<1>)
	{
		lua_pushstring(L, "Split a workgroup into sub groups by common colours (integers).\nExample with mpirun -n 8:\n<pre>"
			
		
"rank = mpi.get_rank()\n"
"size = mpi.get_size()\n"
"\n"
"new_group = {1,1,1,2,2,1,5,2}\n"
"\n"
"split_comm = mpi.comm_split(new_group[rank])\n"
"split_rank = split_comm:get_rank()\n"
"split_size = split_comm:get_size()\n"
"\n"
"for i=1,size do\n"
"	if rank == i then\n"
"		print(rank .. \"/\" .. size .. \" -> \" .. split_rank .. \"/\" .. split_size)\n"
"	end\n"
"	mpi.barrier()\n"
"end\n"
"</pre>Output\n<pre>"
"1/8 -> 1/4\n"
"2/8 -> 2/4\n"
"3/8 -> 3/4\n"
"4/8 -> 1/3\n"
"5/8 -> 2/3\n"
"6/8 -> 4/4\n"
"7/8 -> 1/1\n"
"8/8 -> 3/3\n</pre>"
		);
		lua_pushstring(L, "1 Integer: The colour for the MPI_Comm_split function. Processes with common colours will be put into common sub-workgroups."); 
		lua_pushstring(L, "1 MPI_Comm: Sub-Workgroup");
		return 3;
	}
	
		
//	if(func == &l_mpi_splitrange<0> || func == &l_mpi_splitrange<1>)
	if_FC2(func, l_mpi_splitrange<0>, l_mpi_splitrange<1>)
	{
		lua_pushstring(L, "Make an iterator that iterates over different balanced sequential chunks of a range for each MPI process. Example use:\n"
			"<pre>for i in mpi.range(1,10) do\n\tprint(mpi.get_rank(), i)\nend\n</pre>");
		lua_pushstring(L, "2 Numbers, 1 Optional Number: The first two numbers represent the start and end points (inclusive) of the range. The optional third number gives a step size (default 1)."); 
		lua_pushstring(L, "1 Function: The function will return sequential values in the local chunk for each function call. The function will return nil when the range has been exhausted.");
		return 3;
	}

	return 0;
}

static int l_mpi_metatable(lua_State* L)
{
	lua_newtable(L);
	return 1;
}

static int l_mpi_comm_world(lua_State* L)
{
	lua_pushMPI_Comm(L, MPI_COMM_WORLD);
	return 1;
}

#define add(name, func) \
	lua_pushstring(L, name); \
	lua_pushcfunction(L, func); \
	lua_settable(L, -3); 

void registerMPI(lua_State* L)
{
    luaL_newmetatable(L, "mpi");
    lua_pushstring(L, "__index");
    lua_pushvalue(L, -2);  /* pushes the metatable */
    lua_settable(L, -3);  /* metatable.__index = metatable */
	add("get_size",           l_mpi_get_size<1>   );
	add("get_rank",           l_mpi_get_rank<1>   );
	add("send",               l_mpi_send<1>       );
	add("isend",              l_mpi_isend<1>      );
	add("recv",               l_mpi_recv<1>       );
	add("irecv",              l_mpi_irecv<1>      );
	add("barrier",            l_mpi_barrier<1>    );
	add("gather",             l_mpi_gather<1>     );
	add("bcast",              l_mpi_bcast<1>      );
	add("cart_create",        l_cart_create<1>    );
	add("cart_coord",         l_cart_coord<1>    );
	add("comm_split",         l_mpi_comm_split<1> );
	add("range",              l_mpi_splitrange<1> );
	add("cart_rank",          l_cart_rank<1>      );
	add("__gc",               l_MPI_Comm_gc       );
	add("__tostring",         l_MPI_Comm_tostring );
	lua_pop(L,1); //metatable is registered
	

	lua_newtable(L);
	add("get_processor_name", l_mpi_get_processor_name);
	add("get_size",           l_mpi_get_size<0>       );
	add("get_rank",           l_mpi_get_rank<0>       );
	add("send",               l_mpi_send<0>           );
    add("isend",              l_mpi_isend<0>          );
	add("recv",               l_mpi_recv<0>           );
    add("irecv",              l_mpi_irecv<0>          );
	add("barrier",            l_mpi_barrier<0>        );
	add("gather",             l_mpi_gather<0>         );
	add("bcast",              l_mpi_bcast<0>          );
	add("cart_create",        l_cart_create<0>        );
	add("cart_rank",          l_cart_rank<0>          );
	add("cart_coord",         l_cart_coord<0>         );
	add("comm_split",         l_mpi_comm_split<0>     );
	add("range",              l_mpi_splitrange<0>     );
	add("get_comm_world",     l_mpi_comm_world        );
	add("help",               l_mpi_help              );
	add("metatable",          l_mpi_metatable         );

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


#include "mpi_luafuncs.h"
static int l_getmetatable(lua_State* L)
{
    if(!lua_isstring(L, 1))
        return luaL_error(L, "First argument must be a metatable name");
    luaL_getmetatable(L, lua_tostring(L, 1));
    return 1;
}


MPI_API int lib_register(lua_State* L)
{
	
#ifdef _MPI
	registerMPI(L);
	
	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");
        
        luaL_dofile_mpi_luafuncs(L);

	lua_pushnil(L);
	lua_setglobal(L, "maglua_getmetatable");

	luaT_register<lua_mpi_request>(L);

#endif
	return 0;
}

#include "info.h"
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


