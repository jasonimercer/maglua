#ifndef LUACLIENT
#define LUACLIENT

#include <stdio.h>
#include <string>
#include <map>
#include <vector>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <signal.h>

#include <semaphore.h> 

extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef CLIENT_EXPORTS
  #define CLIENT_API __declspec(dllexport)
 #else
  #define CLIENT_API __declspec(dllimport)
 #endif
#else
 #define CLIENT_API 
#endif


using namespace std;


// typedef void(*pt2FuncVoidStar)(void*);
// typedef void(*pt2FuncCharStar)(char*);

class LuaClient
{
public:
	LuaClient();
	~LuaClient();
	bool connected() {return _connected;};
	bool connectTo(const char* host_port);
	const char* name() {return sourcename.c_str();};
	void disconnect();

	//void uploadLua(lua_State* L);
	int  remoteExecuteLua(lua_State* L);
	int refcount;
private:
	bool _connected;

	string sourcename;

	sem_t addCallbackSem;
	pthread_t updateThread;

	sem_t ioSem;
	bool threadRunning;
	
	
private:
	int sockfd;
	int clientfd;

	int port;
	struct sockaddr_in serv_addr;
	//struct hostent *server;
	
	socklen_t addrlen;
	
	sem_t rwSem;
	sem_t flagSem;
};

LuaClient* checkLuaClient(lua_State* L, int idx);
void lua_pushLuaClient(lua_State* L, LuaClient* lc);
void registerLuaClient(lua_State* L);

#endif

