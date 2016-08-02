#include <iostream>
#include <strings.h>
#include <unistd.h>
#include <error.h>
#include <errno.h>
#include <string.h>

#include "libLuaClient.h"

#define SHUTDOWN        0


extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}

using namespace std;

#define MAXHOSTCACHE 64
static long servers[MAXHOSTCACHE];
static char names[MAXHOSTCACHE][256];
static int num_name_cache = 0;

static int sock_valid(int sockfd)
{
	socklen_t optlen = 0;
	int rv = getsockopt(sockfd, SOL_SOCKET, SO_TYPE, NULL, &optlen);
	
	return rv >= 0;
	
	//return errno != EBADF;	   
}

#ifndef sure_write
#define sure_write(fd, data, sz, ok) sure_write_(fd, data, sz, ok, __FILE__, __LINE__)
#endif
static int sure_write_(int fd, void* data, int sz, int* ok, const char* file, int line)
{
	int b;
	if(!sock_valid(fd))
	{
		*ok = 0;
		return 0;
	}
	
	int msz = write(fd, data, sz);
	while(msz < sz)
	{
		if(!sock_valid(fd))
		{
			*ok = 0;
			return msz;
		}
		b = write(fd, &((char*)data)[msz], sz-msz);
		if(b == -1)
		{
			fprintf(stderr, "write(%i, %lX, %i) error: `%s' (%s:%i)\n", fd, (long)data, sz, strerror(errno), file, line);
			return msz;
		}
		msz += b; 
	}
	return msz;
}

#ifndef sure_read
#define sure_read(fd, data, sz, ok) sure_read_(fd, data, sz, ok, __FILE__, __LINE__)
#endif
static int sure_read_(int fd, void* data, int sz, int* ok, const char* file, int line)
{
	*ok = 1;
	if(!sock_valid(fd))
	{
		*ok = 0;
		return 0;
	}
	int b;
	int msz = read(fd, data, sz);
	while(msz < sz)
	{
		if(!sock_valid(fd))
		{
			*ok = 0;
			return msz;
		}
		
		b = read(fd, &((char*)data)[msz], sz-msz);
		if(b == 0)
		{
			*ok = 0;
			return msz;
		}
		if(b == -1)
		{
			fprintf(stderr, "read(%i, %lX, %i) error: `%s' (%s:%i)\n", fd, (long)data, sz, strerror(errno), file, line);
			return msz;
		}
		msz += b; 
	}
	return msz;
}



LuaClient::LuaClient()
{
	_connected = false;
	refcount = 0;
}

LuaClient::~LuaClient()
{
	disconnect();
}


bool LuaClient::connectTo(const char* host_port)
{
	if(_connected)
		return false;

	char* host = new char[strlen(host_port)+1];
	int port;
	
	strcpy(host, host_port);
	int spot = 0;
	
	for(int i=strlen(host)-1; i>=0; i--)
	{
		if(host[i] == ':')
		{
			host[i] = 0;
			spot = i;
			i = 0;
		}
	}
	
	if(!spot)
	{
		cerr << "Unable to parse host and port: " << host_port << endl;
		delete [] host;
		return false;
	}
	
	port = atoi(host_port + spot + 1);
	
// 	_client = SHM_Connect(host, port);
	
	sockfd = socket(AF_INET, SOCK_STREAM, 0);
	if(sockfd < 0)
	{
		fprintf(stderr, "ERROR: failed to open socket: %s\n", strerror(errno));
		return false;
	}
	
	// gethostbyname seems to have anti-DOS gibbereish inside so we'll cache
	// any lookedup values. This means we'll first look at the cache for
	// server_addys
	long saddr = 0;
	for(int i=0; i<num_name_cache && !saddr; i++)
	{
		if(strcmp(names[i], host) == 0)
		{
			saddr = servers[i];
		}
	}
	
	if(!saddr) //then we need to look it up
	{
		struct hostent *server = gethostbyname(host);
		if(server == NULL) 
		{
			fprintf(stderr,"ERROR: No such host (%s)\n", host);
			return 0;
		}
		
		if(num_name_cache < MAXHOSTCACHE)
		{
			servers[num_name_cache] = *(long*) (server->h_addr_list[0]);
			saddr = servers[num_name_cache];
			strncpy(names[num_name_cache], host, 256);
			num_name_cache++;
		}
	}
	
	memset(&serv_addr, 0, sizeof(serv_addr));	// create and zero struct
	
	serv_addr.sin_family = AF_INET;    /* select internet protocol */
	serv_addr.sin_port = htons(port);         /* set the port # */
	serv_addr.sin_addr.s_addr = saddr; //*(long*) (server->h_addr_list[0]);  /* set the addr */
	
	if(connect(sockfd, (const sockaddr*)&(serv_addr), sizeof(serv_addr)))
	{
		fprintf(stderr, "ERROR: failed to connect to %s:%i\n", host, port);
		fprintf(stderr, "       %s\n", strerror(errno));
		return false;
	}
		
	sem_init(&rwSem, 0, 1);
	sem_init(&ioSem, 0, 1);

	_connected = true;
	sourcename = host_port;

	delete [] host;

	return true;
}

void LuaClient::disconnect()
{
	if(_connected)
	{
		int ok;
		int cmd = SHUTDOWN;
		sem_wait(&rwSem);
		sure_write(sockfd, &cmd, sizeof(int), &ok);
		sure_write(sockfd, &cmd, sizeof(int), &ok);
		//printf("ok: %i\n", ok);
		close(sockfd);
		sockfd = 0;
		sem_post(&rwSem);
		_connected = false;
		//printf("Sent shutdown signal\n");
	}
}


bool LuaClient::rawVarUpload(lua_Variable* v)
{
	int ok = 1;
	int v3[3];
	v3[0] = v->chunklength;
	v3[1] = v->listlength;
	v3[2] = v->type;
	
	sure_write(sockfd, v3, sizeof(int)*3, &ok); if(!ok) return false;

	if(v->chunklength)
	{
		sure_write(sockfd, v->chunk, v->chunklength, &ok); 
		if(!ok) return false;
	}

	if(v->listlength)
	{
		for(int i=0; i<v->listlength; i++)
		{
			if(!rawVarUpload(&v->listKey[i])) return false;
			if(!rawVarUpload(&v->listVal[i])) return false;
		}
	}
	return true;
}


bool LuaClient::rawVarDownload(lua_Variable* v)
{
	int ok = 1;
	initLuaVariable(v);
	int v3[3];
	
	sure_read(sockfd, v3, sizeof(int)*3, &ok);
	if(!ok) return false;
	
	v->chunksize = v3[0];
	v->chunklength = v3[0];
	v->listlength = v3[1];
	v->type = v3[2];
	
	if(v->chunksize)
	{
		v->chunk = (char*)malloc(sizeof(char)*v->chunklength);
		sure_read(sockfd, v->chunk, v->chunklength, &ok); 
		if(!ok) return false;
	}

	if(v->listlength)
	{
		v->listKey = (lua_Variable*)malloc(sizeof(lua_Variable)*v->listlength);
		v->listVal = (lua_Variable*)malloc(sizeof(lua_Variable)*v->listlength);
		for(int i=0; i<v->listlength; i++)
		{
			if(!rawVarDownload(&v->listKey[i])) return false;
			if(!rawVarDownload(&v->listVal[i])) return false;
		}
	}
	return true;
}

#if 0
void LuaClient::uploadLua(lua_State* L)
{
	int ok = 1;
	int n = lua_gettop(L);

	if(!n) return;

 	sem_wait(&ioSem);
	lua_Variable* vars = (lua_Variable*)malloc(sizeof(lua_Variable)*n);

	for(int i=0; i<n; i++)
	{
		initLuaVariable(&vars[i]);
		exportLuaVariable(L, i+1, &vars[i]);
	}

	int b[2];
	b[0] = 10; //10 == UPLOADLUA
	b[1] = n;

	sure_write(sockfd, b, sizeof(int)*2);

	for(int i=0; i<n; i++)
		rawVarUpload(&vars[i]);
	
	for(int i=0; i<n; i++)
		freeLuaVariable(&vars[i]);

	free(vars);
 	sem_post(&ioSem);
}
#endif

int LuaClient::remoteExecuteLua(lua_State* L)
{
	int ok = 1;
 	sem_wait(&ioSem);
	int n = lua_gettop(L);

	if(!n)
	{
		sem_post(&ioSem);
		return 0;
	}

	lua_Variable* vars = (lua_Variable*)malloc(sizeof(lua_Variable)*n);

	
	for(int i=0; i<n; i++)
	{
		//printf("%i/%i\n", i+1, n);
		initLuaVariable(&vars[i]);
		exportLuaVariable(L, i+1, &vars[i]);
		//printf("%i/%i\n", i+1, n);
	}

	int b[2];
	b[0] = 11; //11 == REMOTELUA
	b[1] = n;

	sure_write(sockfd, b, sizeof(int)*2, &ok);
	if(!ok)
	{
		for(int i=0; i<n; i++)
			freeLuaVariable(&vars[i]);
		free(vars);
		sem_post(&ioSem); //we can let other communication happen now. 
		return luaL_error(L, "network fail in :remote");
	}

	for(int i=0; i<n; i++)
	{
		if(!rawVarUpload(&vars[i]))
		{
			for(int i=0; i<n; i++)
				freeLuaVariable(&vars[i]);
			free(vars);
			sem_post(&ioSem); //we can let other communication happen now. 
			return luaL_error(L, "network fail in :remote");
		}

		//printf("up %i/%i\n", i+1, n);
	}
	
	//lua function is now being run on the server. 
 	sem_post(&ioSem); //we can let other communication happen now. 

	for(int i=0; i<n; i++)
		freeLuaVariable(&vars[i]);
	free(vars);

	//the server has "consumed" the variables. we should pop them from the stack.
	lua_pop(L, n);

 	sem_wait(&ioSem);
	sure_read(sockfd, &n, sizeof(int)*1, &ok); //get return count
	if(!ok)
	{
		sem_post(&ioSem);
		return luaL_error(L, "network fail in :remote");
	}
	
	//printf("%i return values\n", n);
	
	//n return variables
	vars = (lua_Variable*)malloc(sizeof(lua_Variable)*n);
	for(int i=0; i<n; i++)
	{
		if(!rawVarDownload(&vars[i]))
		{
			for(int j=0; j<i-1; j++)
				freeLuaVariable(&vars[j]);
			free(vars);
			sem_post(&ioSem);
			return luaL_error(L, "network fail in :remote");
		}
	}
	sem_post(&ioSem);
	
	//now to put the downloaded vars on the stack
	for(int i=0; i<n; i++)
	{
		importLuaVariable(L, &vars[i]);
		freeLuaVariable(&vars[i]);
	}

	free(vars);
	return n;
}













LuaClient* checkLuaClient(lua_State* L, int idx)
{
	LuaClient** pp = (LuaClient**)luaL_checkudata(L, idx, "Client");
	luaL_argcheck(L, pp != NULL, 1, "Client' expected");
	return *pp;
}

void lua_pushLuaClient(lua_State* L, LuaClient* lc)
{
	lc->refcount++;
	LuaClient** pp = (LuaClient**)lua_newuserdata(L, sizeof(LuaClient**));
	*pp = lc;
	luaL_getmetatable(L, "Client");
	lua_setmetatable(L, -2);
}


static int l_client_remote(lua_State* L)
{
	LuaClient* lc = checkLuaClient(L, 1);
	lua_remove(L, 1); //get object off the stack
	return lc->remoteExecuteLua(L);
}


static int l_client_gc(lua_State* L)
{
	LuaClient* lc = checkLuaClient(L, 1);
	lc->refcount--;
	if(lc->refcount <= 0)
	{
		delete lc;
	}
	return 0;
}
static int l_client_tostring(lua_State* L)
{
	LuaClient* lc = checkLuaClient(L, 1);
	lua_pushstring(L, "Client");
	return 1;
}

static int l_client_connect(lua_State* L)
{
	LuaClient* lc = checkLuaClient(L, 1);
	lua_pushboolean(L, lc->connectTo(lua_tostring(L, 2)));
	return 1;
}

static int l_client_connected(lua_State* L)
{
	LuaClient* lc = checkLuaClient(L, 1);
	lua_pushboolean(L, lc->connected());
	return 1;
}

static int l_client_disconnect(lua_State* L)
{
	LuaClient* lc = checkLuaClient(L, 1);
	lc->disconnect();
	return 0;
}

static int l_client_new(lua_State* L)
{
	LuaClient* lc = new LuaClient;
	
	if(lua_isstring(L, 1))
	{
		if(!lc->connectTo(lua_tostring(L, 1)))
		{
			delete lc;
			return luaL_error(L, "Failed to connect to `%s'", lua_tostring(L, 1));
		}
	}
	
	lua_pushLuaClient(L, lc);
	return 1;
}


void registerLuaClient(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
	{"__gc",         l_client_gc},
	{"__tostring",   l_client_tostring},
	{"remote",       l_client_remote},
	{"connect",      l_client_connect},
	{"connected",    l_client_connected},
	{"disconnect",   l_client_disconnect},
	{NULL, NULL}
	};
	
	luaL_newmetatable(L, "Client");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
	
	static const struct luaL_reg functions [] = {
		{"new",                 l_client_new},
		{NULL, NULL}
	};
	
	luaL_register(L, "Client", functions);
	lua_pop(L,1);
}

int lib_register(lua_State* L)
{
	registerLuaClient(L);
	return 0;
}
