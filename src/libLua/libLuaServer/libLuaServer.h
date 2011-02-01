#ifndef LUA_SERVER
#define LUA_SERVER

#include <stdio.h>
#include <semaphore.h>
#include <stdlib.h>
#include <iostream>
#include <queue>
#include <pthread.h> 
#include <vector>
#include <deque>
#include <map>
#include <setjmp.h>

#include "libLuaMigrate.h"

using namespace std;

extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}

class LuaComm
{
public:
	LuaComm(const char* Name);
	~LuaComm();
	void do_comm(int fd);

	void addError (string e);
	void addInfo  (string e);

	const char* name();

private:
	void rawVarRead(int fd, lua_Variable* v);
	void rawVarWrite(int fd, lua_Variable* v);

	sem_t qSem;
	queue<string> errors;
	queue<string> infos;

	char _name[512];
};

struct LuaThread
{
	pthread_t thread;
	lua_State* L;
	LuaComm* comm;
	string fingerprint;
	jmp_buf env;
	int id;
	sem_t sem;
};

typedef int(*luafunc)(lua_State*);

#define LuaServer (LuaServer_::Instance())

class LuaServer_
{
public:
	static LuaServer_& Instance()
	{
		static LuaServer_ theServer;
		return theServer;
	}

	bool init(int argc, char** argv);
	void serve();

	void addComm(LuaComm* comm);
	void removeComm(LuaComm* comm);

	void addError(string e, LuaComm* c=0);
	void addInfo (string e, LuaComm* c=0);

	int  establish(unsigned short portnum);
	int  get_connection(int s, char* c);

	int addLuaFunction(lua_Variable* vars, int nvars, LuaComm* comm);
 	lua_Variable* executeLuaFunction(lua_Variable* vars, int nvars, LuaComm* comm, int* nret);

	LuaThread* getThread(int pid);
	
	static int port;
	static luafunc registerCallback;

	static vector<pthread_t*> commThread;

	static pthread_t rootThread;

	static sem_t luaThreadSem;
	static int nextThreadID;
	static vector<LuaThread*> runningLuathreads;

	static sem_t addCommSem;
	static sem_t updateSem;

	static vector<LuaComm*> comms;
};



#endif

