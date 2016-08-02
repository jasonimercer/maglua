#include "webserver.h"

#include<stdio.h>
#include<string.h>    //strlen
#include<stdlib.h>    //strlen
#include<sys/socket.h>
#include<arpa/inet.h> //inet_addr
#include<unistd.h>    //write
#include<pthread.h> //for threading , link with lpthread

#define MESSAGE_SIZE (1024*8)


WebServer::WebServer()
    : LuaBaseObject(hash32(lineage(0)))
{
    init();
}

WebServer::~WebServer()
{
    deinit();
}

void WebServer::init()
{
    port = 8888;
    data_ref = LUA_REFNIL;
    ref_client_function = LUA_REFNIL;
    sem_init(&mutex, 0, 1);
}

void WebServer::deinit()
{
    luaL_unref(L, LUA_REGISTRYINDEX, data_ref);
    luaL_unref(L, LUA_REGISTRYINDEX, ref_client_function);
    data_ref = LUA_REFNIL;
    ref_client_function = LUA_REFNIL;
    sem_destroy(&mutex);
}

int WebServer::getInternalData(lua_State* L)
{
    lua_rawgeti(L, LUA_REGISTRYINDEX, data_ref);
    return 1;
}

void WebServer::setInternalData(lua_State* L, int stack_pos)
{
    lua_pushvalue(L, stack_pos);
    int dr = luaL_ref(L, LUA_REGISTRYINDEX);
    luaL_unref(L, LUA_REGISTRYINDEX, data_ref);
    data_ref = dr;
}



void WebServer::lock()
{
    sem_wait(&mutex);
}
void WebServer::unlock()
{
    sem_post(&mutex);
}

int WebServer::luaInit(lua_State* L, const int base)
{
    LuaBaseObject::luaInit(L);

    


    return 0;
}

static int l_writeback(lua_State* L)
{
    int sock = lua_tointeger(L, lua_upvalueindex(1));

    if(lua_isstring(L, 1))
    {
        const char* client_message = lua_tostring(L, 1);
        
        int w = write(sock , client_message , strlen(client_message)+1);
    }
    
    return 0;
}


int WebServer::handleClientMessage(const char* msg, int session_id, int sock_fd)
{
    if(ref_client_function == LUA_REFNIL)
        return 0;

    lock();

    lua_rawgeti(L, LUA_REGISTRYINDEX, ref_client_function);
    luaT_push<WebServer>(L, this);
    lua_pushinteger(L, session_id);
    lua_pushstring(L, msg);

    lua_pushinteger(L, sock_fd);
    lua_pushcclosure(L, l_writeback, 1);

    lua_call(L, 4, 1);

    int i = lua_tointeger(L, -1);
    lua_pop(L, 1);
    unlock();

    return i;
}


void WebServer::encode(buffer* b)
{
    ENCODE_PREAMBLE;
        
}

int WebServer::decode(buffer* b)
{

    return 0;
}

typedef struct WS_struct
{
    WS_struct(WebServer* _ws, int sd, int sid)
        {
            ws = _ws;
            socket_desc = sd;
            session_id = sid;
        }
    int socket_desc;
    int session_id;
    WebServer* ws;
} WS_struct;

void* connection_handler(void* _wss)
{
    //Get the socket descriptor
    WS_struct* wss = (WS_struct*)_wss;
    int sock = wss->socket_desc;
    WebServer* ws = wss->ws;
    int session_id = wss->session_id;

    delete wss; // got the data

    int read_size;
    char client_message[MESSAGE_SIZE];
     
    //Receive a message from client
    while( (read_size = recv(sock, client_message, MESSAGE_SIZE-1, 0)) > 0 )
    {
        //end of string marker
        client_message[read_size] = '\0';

        ws->handleClientMessage(client_message, session_id, sock);

        /*
        //Send the message back to client
        int w = write(sock , client_message , strlen(client_message));
        
        //clear the message buffer
        memset(client_message, 0, MESSAGE_SIZE);
        */
    }
     
    if(read_size == 0)
    {
        fflush(stdout);
    }
    else if(read_size == -1)
    {
        perror("recv failed");
    }
         
    return 0;
} 


int WebServer::main(lua_State* L)
{
    int socket_desc , client_sock , c;
    struct sockaddr_in server , client;
     
    //Create socket
    socket_desc = socket(AF_INET , SOCK_STREAM , 0);
    if (socket_desc == -1)
    {
        return luaL_error(L, "Could not create socket");
    }
     
    //Prepare the sockaddr_in structure
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = htons( port );
     
    //Bind
    if( bind(socket_desc,(struct sockaddr *)&server , sizeof(server)) < 0)
    {
        return luaL_error(L, "Bind Failed");
    }
     
    //Listen
    listen(socket_desc , 3);
     
    //Accept and incoming connection
    c = sizeof(struct sockaddr_in);
     
     
    //Accept and incoming connection
    c = sizeof(struct sockaddr_in);
    pthread_t thread_id;
    int session_id = 1;

    while( (client_sock = accept(socket_desc, (struct sockaddr *)&client, (socklen_t*)&c)) )
    {
        if( pthread_create( &thread_id, NULL,  connection_handler, (void*)new WS_struct(this, client_sock, session_id)) < 0)
        {
            return luaL_error(L, "could not create thread");
        }
         
        session_id++;
    }
     
    if (client_sock < 0)
    {
        return luaL_error(L, "Accept Failed");
    }
     
    return 0;
}
 

static int l_setport(lua_State* L)
{
    LUA_PREAMBLE(WebServer, ws, 1);
    ws->port = lua_tointeger(L, 2);
    return 0;
}

static int l_getport(lua_State* L)
{
    LUA_PREAMBLE(WebServer, ws, 1);
    lua_pushinteger(L, ws->port);
    return 1;
}


static int l_start(lua_State* L)
{
    LUA_PREAMBLE(WebServer, ws, 1);
    return ws->main(L);
}


static int l_scf(lua_State* L)
{
    LUA_PREAMBLE(WebServer, ws, 1);

    if(ws->ref_client_function != LUA_REFNIL)
        luaL_unref(L, LUA_REGISTRYINDEX, ws->ref_client_function);

    ws->ref_client_function = luaL_ref(L, LUA_REGISTRYINDEX);
    
    return ws->main(L);
}

static int l_getinternaldata(lua_State* L)
{
    LUA_PREAMBLE(WebServer, ws, 1);
    return ws->getInternalData(L);
}
static int l_setinternaldata(lua_State* L)
{
    LUA_PREAMBLE(WebServer, ws, 1);
    ws->setInternalData(L, 2);
    return 0;
}



int WebServer::help(lua_State* L)
{
    if(lua_gettop(L) == 0)
    {
        lua_pushstring(L, "WebServer");
        lua_pushstring(L, "Constructor arguments");
        lua_pushstring(L, ""); //output, empty
        return 3;
    }

    lua_CFunction func = lua_tocfunction(L, 1);

    return LuaBaseObject::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* WebServer::luaMethods()
{
    if(m[127].name)return m;
    merge_luaL_Reg(m, LuaBaseObject::luaMethods());

    static const luaL_Reg _m[] =
	{
            {"_start",        l_start},
            {"_setClientFunction", l_scf},
            {"_setPort", l_setport},
            {"_port", l_getport},
            {"_getinternaldata", l_getinternaldata},
            {"_setinternaldata", l_setinternaldata},
            {NULL, NULL}
	};
    merge_luaL_Reg(m, _m);
    m[127].name = (char*)1;
    return m;
}



#include "info.h"

#include "webserver_luafuncs.h"
static int l_getmetatable(lua_State* L)
{
    if(!lua_isstring(L, 1))
        return luaL_error(L, "First argument must be a metatable name");
    luaL_getmetatable(L, lua_tostring(L, 1));
    return 1;
}

extern "C"
{
int lib_register(lua_State* L)
{
	luaT_register<WebServer>(L);

        // augmenting metatable with custom lua code
        lua_pushcfunction(L, l_getmetatable);
        lua_setglobal(L, "maglua_getmetatable");

        luaL_dofile_webserver_luafuncs(L);

        lua_pushnil(L);
        lua_setglobal(L, "maglua_getmetatable");
	return 0;
}

int lib_version(lua_State* L)
{
    return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "WebServer";
#else
	return "WebServer-Debug";
#endif
}

int lib_main(lua_State* L)
{
	return 0;
}
}
