#include "luamigrate.h"
#include "libLuaServer.h"
#include "net_helpers.h"

// #include "main.h"
#include "libMagLua.h"
#include "luabaseobject.h"
#include "import.h"

#include <stdio.h>
#include <semaphore.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <queue>
#include <pthread.h> 
#include <vector>
#include <setjmp.h>
#include <errno.h>
#include <sys/types.h>
#include <string.h>

#ifndef WIN32
#include <unistd.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <strings.h>
#include <error.h>
#else
 #pragma warning(disable: 4251)
 #pragma warning(disable: 4996)
 #include <WinSock.h>
 typedef int socklen_t;
 static int WSAStartupCalled = 0;
// #define write(a,b,c) send(a,(const char*)b,c,0)
// #define read(a,b,c) recv(a,(char*)b,c,0)
 #define close(a) closesocket(a)

 //SD_BOTH = 2
 #define SHUT_RDWR 2
#endif

#define SHUTDOWN        0
#define UPLOADLUA       10
#define REMOTELUA       11


extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}

using namespace std;

static int registerMagLuaMain(lua_State* L);
static int copy_registry(lua_State* src, lua_State* dest);

typedef int(*luafunc)(lua_State*);





class LuaServer;

class LuaComm
{
public:
	struct CommData
	{
		int fd;
		string name;
		int status;
		pthread_t* thread;
		LuaServer* server;
	};
	
	LuaComm(const char* Name, CommData* d);
	~LuaComm();
	void do_comm(int fd);

	void addError (string e);
	void addInfo  (string e);

	const char* name();

	CommData* cdata;
	LuaServer* server;
	
private:
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


class LuaServer : public LuaBaseObject
{
public:
	LuaServer(int port=55000);
	~LuaServer();
	
	LINEAGE1("Server")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	
	void serve();
	void servebg();
	void addComm(LuaComm* comm);
	int removeComm(/*LuaComm* comm*/); //remove any dead comms (and pthread_join them)

	void addError(string e, LuaComm* c=0);
	void addInfo (string e, LuaComm* c=0);

	int  establish(unsigned short portnum);
	int  get_connection(int s, char* c);

// 	int addLuaFunction(lua_Variable* vars, int nvars, LuaComm* comm);
 	void executeLuaFunction(LuaComm* comm, LuaVariableGroup& input, LuaVariableGroup& output);

	LuaThread* getThread(int pid);
	
	int port;
	
	vector<string> share_list;
	lua_State* server_L;
	lua_State* holder_L;

	pthread_t rootThread;
	pthread_t bgThread;
	
	sem_t luaThreadSem;
	int nextThreadID;
	vector<LuaThread*> runningLuathreads;

	sem_t addCommSem;
	sem_t updateSem;

	sem_t holderSem;

	sem_t availableStates;

	vector<LuaComm*> comms;
	
	jmp_buf root_env;
};








void __threadSignalHandler(int signo);




void* __threadCommMain(void* args)
{
	LuaComm::CommData* d = (LuaComm::CommData*)args;

	int   fd =   d->fd;
	LuaComm* mc = new LuaComm(d->name.c_str(),  d);

	d->server->addComm(mc);
	
	mc->do_comm(fd);

	//printf(">>>> %s:%i shutdown\n", __FILE__, __LINE__);
	shutdown(fd, SHUT_RDWR);
	//printf(">>>> %s:%i close\n", __FILE__, __LINE__);
	close(fd);

	d->status = 0;
	
	pthread_exit(0);
	return 0;
}

void __main_kill(int)
{
// 	if(pthread_equal(LuaServer.rootThread, pthread_self()))
// 	{
// 		longjmp(root_env, 1);
// 	}
}

LuaServer::LuaServer(int default_port)
	: LuaBaseObject(hash32(LuaServer::typeName()))
{
#ifdef WIN32
	if(!WSAStartupCalled)
	{
		WSAStartupCalled = 1;
		WORD wVersionRequested;
		WSADATA wsaData;
		wVersionRequested = MAKEWORD(1, 1);
		int err = WSAStartup(wVersionRequested, &wsaData); 
		if (err != 0)
	        printf("(%s:%i) WSAStartup error: %i\n", __FILE__, __LINE__, err);
	}
#endif
	sem_init(&addCommSem,   0, 1);
	sem_init(&updateSem,    0, 1);
	sem_init(&luaThreadSem, 0, 1);
	sem_init(&addCommSem,   0, 1);

	sem_init(&availableStates,  0, 10);
	
	refcount = 0;
	port = default_port;
	server_L = 0;
	rootThread = pthread_self();
	
	sem_init(&holderSem, 0, 1);
	holder_L = lua_open();
	registerMagLuaMain(holder_L);

//	registerCallback = 0;
}


int LuaServer::luaInit(lua_State* _L)
{
	L = _L;
	if(lua_isnumber(L, 1))
		port = lua_tointeger(L, 1);
	return 0;
}


void LuaServer::push(lua_State* L)
{
	luaT_push<LuaServer>(L, this);
}

	
LuaServer::~LuaServer()
{
	//lua_close(server_L);
	lua_close(holder_L);
}


void LuaServer::addError(string e, LuaComm* c)
{
	sem_wait(&updateSem);
	sem_wait(&addCommSem);

	if(c)
		c->addError(e);
	else
		for(unsigned int i=0; i<comms.size(); i++)
			comms[i]->addError(e);

	sem_post(&addCommSem);
	sem_post(&updateSem);
}
void LuaServer::addInfo(string e, LuaComm* c)
{
	sem_wait(&updateSem);
	sem_wait(&addCommSem);

	if(c)
		c->addInfo(e);
	else
		for(unsigned int i=0; i<comms.size(); i++)
			comms[i]->addInfo(e);

	sem_post(&addCommSem);
	sem_post(&updateSem);
}


static void* __threadBG(void* args)
{
	LuaServer* ls = (LuaServer*)args;

	ls->serve();
	
	pthread_exit(0);
	return 0;
}


void LuaServer::servebg()
{
	pthread_create(&bgThread, NULL, __threadBG, (void *)this);
}


void LuaServer::serve()
{
	int fd, s;
	if((s = establish(port)) < 0)
 	{
		perror("establish");
		exit(1);
	}

	char connectionName[512];

	int r = setjmp(root_env);

	if(!r)
	for(;;)
	{
		fd = get_connection(s, connectionName);
		if(fd < 0)
		{
			fprintf(stderr, "ERROR: %s\n", strerror(errno));
			if(errno == EINTR) /* EINTR might happen on accept(), */
				continue; /* try again */
			perror("accept"); /* bad */
				exit(1);
		}

		pthread_t* thread = new pthread_t();

		LuaComm::CommData* a = new LuaComm::CommData;
		a->fd = fd;
		a->name = connectionName;
		a->thread = thread;
		a->server = this;
// 		commThread.push_back(thread);

		pthread_create(thread, NULL, __threadCommMain, (void *)a);
		
		while(removeComm());
	}
	
	while(removeComm());
}

int LuaServer::establish(unsigned short portnum)
{
	char myname[128+1];
	int s;
	struct sockaddr_in sa;
	struct hostent *hp;
	
	memset(&sa, 0, sizeof(struct sockaddr_in)); 
	gethostname(myname, 128);
	strcpy(myname, "localhost"); //some computer's hostname don't resolve to 127.0.0.1
	hp = gethostbyname(myname);
	if (hp == NULL)
	{
#ifdef WIN32
		int err = WSAGetLastError();
		fprintf(stderr, "gethostbyname. Error = %i\n", err);
#else
		perror("gethostbyname");
#endif
		return(-1);
	}
	sa.sin_family= hp->h_addrtype; /* this is our host address */
	sa.sin_port= htons(portnum); /* this is our port number */

	//printf(">>>> %s:%i socket\n", __FILE__, __LINE__);
	if((s= socket(AF_INET, SOCK_STREAM, 0)) < 0) /* create socket */
	{
		perror("socket");
		return(-1);
	}
	//printf(">>>> %s:%i bind\n", __FILE__, __LINE__);
	if(bind(s,(struct sockaddr *)&sa,sizeof(struct sockaddr_in)) < 0) 
	{
		close(s);
		perror("bind");
		return(-1); /* bind address to socket */
	}

	//printf(">>>> %s:%i listen\n", __FILE__, __LINE__);
	listen(s, 128); /* max # of queued connects */
	return(s);
}

int LuaServer::get_connection(int s, char* connectionName)
{
	sockaddr_in cli_addr;
	socklen_t addrlen = 1024;

	int fd;
	printf("waiting for connection\n");
	//printf(">>>> %s:%i accept\n", __FILE__, __LINE__);
	fd = accept(s, (struct sockaddr*)(&cli_addr), &addrlen);
	if(fd < 0)
	{
		fprintf(stderr, "ERROR: %s\n", strerror(errno));
		return(-1);
	}

	//cout << "ACCEPT " << fd << endl;

	printf("new connection from %s:%i\n", inet_ntoa( cli_addr.sin_addr), cli_addr.sin_port);
	sprintf(connectionName, "%s:%i", inet_ntoa( cli_addr.sin_addr), cli_addr.sin_port);
	
	return fd;
}

void LuaServer::addComm(LuaComm* comm)
{
	sem_wait(&addCommSem);
		comms.push_back(comm);	
	sem_post(&addCommSem);
}


// remove comm if it is dead
int LuaServer::removeComm()
{
	sem_wait(&addCommSem);
	bool rem = false;
	vector<LuaComm*>::iterator it;

	for(it = comms.begin();it != comms.end(); ++it)
	{
		if( (*it)->cdata->status == 0 )
		{
			rem = true;
			break;
		}
	}

	if(rem)
	{
		// cout << "PTHREAD_JOIN" << endl;
		if(pthread_join(*(*it)->cdata->thread, NULL))
			cerr << "PTHREAD JOIN ERROR" << endl;

		delete (*it)->cdata->thread;
		delete (*it)->cdata;
		delete *it;
		comms.erase(it);
	}

	sem_post(&addCommSem);
	
	return rem;
}

void threadShutdown()
{
#if 0
#ifdef WIN32
#warning need look at this
#endif
	sem_wait(&LuaServer.luaThreadSem);
	vector<LuaThread*>::iterator it;

	for(it=LuaServer.runningLuathreads.begin(); it != LuaServer.runningLuathreads.end(); ++it)
	{
		if(pthread_equal( (*it)->thread, pthread_self() ))
		{
			sem_post(&LuaServer.luaThreadSem);
			longjmp((*it)->env, 1);
		}
	}
	sem_post(&LuaServer.luaThreadSem);

	fprintf(stderr, "FAILED to find self in thread vector (%s:%i)\n", __FILE__, __LINE__);
#endif
}

void __threadSignalHandler(int signo)
{
#if 0
#ifdef WIN32
#warning need look at this
#endif
	if(signo == SIGUSR1)
	{
		if(!pthread_equal(LuaServer.rootThread, pthread_self()))
		{
			//printf("SIGUSR1 - thread long jumping\n");
			threadShutdown();
		}
		else
		{
			//printf("root got SIGUSR1, ignoring\n");
		}
	}
#endif
}

LuaThread* LuaServer::getThread(int pid)
{
	LuaThread* t = 0;
	sem_wait(&luaThreadSem);
	vector<LuaThread*>::iterator it;

	for(it=runningLuathreads.begin(); it != runningLuathreads.end(); ++it)
	{
		if((*it)->id == pid)
		{
			t = *it;
			break;
		}
	}
	sem_post(&luaThreadSem);
	return t;
}

void* __threadLuaMain(void* args)
{
#ifndef WIN32
	signal(SIGUSR1, __threadSignalHandler); //install sig handler
#endif

	LuaThread* thread = (LuaThread*)args;

	lua_State*  L = thread->L;
	LuaComm* c = thread->comm;

	int r = setjmp(thread->env);

	if(!r)
	{
		int nargs = lua_gettop(L)-1;
	
		if(lua_pcall(L, nargs, 0, 0))
		{
			string s = lua_tostring(L, -1);
			cerr << s << endl;
			c->server->addError(s, c); //only error back to src client, not all clients
		}
	}

	sem_t& sem = c->server->luaThreadSem;
	

	sem_wait(&sem);
	vector<LuaThread*>::iterator it;

	for(it=c->server->runningLuathreads.begin(); it != c->server->runningLuathreads.end(); ++it)
	{
		if(pthread_equal( (*it)->thread, pthread_self() ))
		{
			LuaThread* lv = *it;
			c->server->runningLuathreads.erase(it);

			lua_close(lv->L);
			delete lv;

			break;
		}
	}
	sem_post(&sem);
	return 0;
}


#if 0
int LuaServer::addLuaFunction(LuaComm* comm, LuaVariableGroup& group)
{
	int retval = -1;
	lua_State* L;
	L = lua_open();
	luaL_openlibs(L);
	if(registerCallback)
		registerCallback(L);
	
	for(int i=0; i<nvars; i++)
		importLuaVariable(L, &vars[i]);

	LuaThread* thread = new LuaThread();
	thread->L = L;
	thread->comm = comm;

	sem_init(&(thread->sem), 0, 1);

	sem_wait(&luaThreadSem);
		thread->id = nextThreadID;
		retval = thread->id;
		nextThreadID++;
		runningLuathreads.push_back(thread);
	sem_post(&luaThreadSem);

	pthread_attr_t tattr;
	int ret;
	/* set the thread detach state */
	ret = pthread_attr_setdetachstate(&tattr,PTHREAD_CREATE_DETACHED);
	
	pthread_create(&(thread->thread), &tattr, __threadLuaMain, (void *)thread);
	pthread_detach(thread->thread); //we wont be joining
	return retval;
}
#endif

static int registerMagLuaMain(lua_State* L)
{
	int sub_process = 1;
	int force_quiet = 1;
	//MagLua_set_and_run(L, sub_process);
	libMagLua(L, sub_process, force_quiet);

	
#if 0
#ifndef WIN32
	return registerMain(L);
#else
	typedef int (*func)(lua_State*);
	func registerMain = import_function<func>("", "registerMain");
	if(!registerMain)
	{
		fprintf(stderr, "Failed to find `registerMain'\n");
	}
	else
	{
		return registerMain(L);
	}
	return 0;
#endif
#endif
	
	return 0;
}

void LuaServer::executeLuaFunction(LuaComm* comm, LuaVariableGroup& input, LuaVariableGroup& output)
{
	//limit the number of simultaneous lua states running
	//this can cause a problem if the server runs code that
	//makes a new connection to itself and that connection cannot run
	sem_wait(&availableStates);
	
	lua_State* L;
	L = lua_open();
	luaL_openlibs(L);
	
	registerMagLuaMain(L);

	copy_registry(holder_L, L);
	
	for(unsigned int i=0; i<share_list.size(); i++)
	{
		lua_getglobal(holder_L, share_list[i].c_str());
		if(lua_isuserdata(holder_L, -1))
		{
			LuaBaseObject** p_lbo = (LuaBaseObject**)lua_touserdata(holder_L, -1);
			LuaBaseObject* lbo = *p_lbo;
			lbo->push(L);
			lua_pop(holder_L, 1);
		}
		else
		{
			lua_xmove(holder_L, L, 1);
		}
		lua_setglobal(L, share_list[i].c_str());
	}
	sem_post(&holderSem);
	
// 	if(registerCallback)
// 		registerCallback(L);

	input.writeState(L);
	
	int nargs = lua_gettop(L)-1;

	if(!lua_isfunction(L, 1))
	{
		if(lua_isstring(L, 1))
			printf("Error: %s\n", lua_tostring(L, 1));
	}
	else
	if(lua_pcall(L, nargs, LUA_MULTRET, 0))
	{
		string s = lua_tostring(L, -1);
		cerr << s << endl;
		addError(s, comm); //only error back to src client, not all clients
	}

	output.readState(L);

	lua_close(L);

	sem_post(&availableStates);
}



LuaComm::LuaComm(const char* Name, LuaComm::CommData* d)
{
	sem_init(&qSem, 0, 1);
	strncpy(_name, Name, 512);
	cdata = d;
	cdata->status = 1;
	server = d->server;
}

LuaComm::~LuaComm()
{
}

const char* LuaComm::name()
{
	return _name;
}

void LuaComm::addError (string e)
{
	sem_wait(&qSem);
	errors.push(e);
	sem_post(&qSem);	
}
void LuaComm::addInfo (string e)
{
	sem_wait(&qSem);
	infos.push(e);
	sem_post(&qSem);	
}


void LuaComm::do_comm(int fd)
{
 	//cout << "DO COMM " << fd << endl;
	int loop = 1;
	bool ok = true;
	int cmd;

	LuaVariableGroup input;
	LuaVariableGroup output;
	
	while(loop && ok)
	{
		const int b = sure_read(fd, &cmd, sizeof(int), &ok); 
		
		if(!ok)
		{
			continue; //and exit loop
		}

		if(b < sizeof(int))
			cmd = SHUTDOWN;
	
		switch(cmd)
		{
			case SHUTDOWN:
				loop = 0;
			break;
		
			#if 0
			//not supporting upload right now
			case UPLOADLUA:
				//lua_Variable* luavars;
				numluavars = j;
				luavars = (lua_Variable*)malloc(sizeof(lua_Variable) * (numluavars+1));

				for(i=0; i<numluavars; i++)
				{
					if(!rawVarRead(fd, &luavars[i]))
					{
						ok = 0; continue;
					}
				}
				
				//add it to a state
				LuaServer.addLuaFunction(luavars, numluavars, this);

				for(i=0; i<numluavars; i++)
					freeLuaVariable(&luavars[i]);
				free(luavars);
			break;
			#endif

			case REMOTELUA:
			{
				input.read(fd, ok);
				server->executeLuaFunction(this, input, output);
				input.clear();
				output.write(fd, ok);
				output.clear();
			}
			break;
		}
	}

	cout << "Client disconnect" << endl;
}























static int l_start(lua_State* L)
{
	LUA_PREAMBLE(LuaServer, ls, 1);
	ls->serve();
	return 0;
}


static int l_startbg(lua_State* L)
{
	LUA_PREAMBLE(LuaServer, ls, 1);
	ls->servebg();
	return 0;
}

static int copy_registry(lua_State* src, lua_State* dest)
{
	lua_pushnil(src);
	while(lua_next(src, LUA_REGISTRYINDEX) != 0)
	{
		if(lua_istable(src, -1))
		{
			lua_pushvalue(src, -2);
			lua_pushvalue(src, -2);
			
			lua_xmove(src, dest, 2);
			lua_settable(dest, LUA_REGISTRYINDEX);
		}
			
       lua_pop(src, 1);
     }
     return 0;
}

static int l_share(lua_State* L)
{
	LUA_PREAMBLE(LuaServer, ls, 1);

	sem_wait(&(ls->holderSem));

	copy_registry(L, ls->holder_L);

	for(int i=2; i<=lua_gettop(L); i++)
	{
		int stack_size = lua_gettop(L);
		const char* name = lua_tostring(L, i);
		lua_getglobal(L, name);
		
		if(lua_isuserdata(L, -1))
		{
			LuaBaseObject** p_lbo = (LuaBaseObject**)lua_touserdata(L, -1);
			LuaBaseObject* lbo = *p_lbo;
			lbo->push(ls->holder_L);
			lua_pop(L, 1);
		}
		else
		{
			//printf("moving over %s\n", lua_typename(L, lua_type(L, -1)));
			lua_xmove(L, ls->holder_L, 1);
		}
		lua_setglobal(ls->holder_L, name);
		ls->share_list.push_back(name);
		while(lua_gettop(L) > stack_size)
			lua_pop(L, 1);
	}
	sem_post(&(ls->holderSem));
	return 0;
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* LuaServer::luaMethods()
{
	if(m[127].name)return m;

	static const luaL_Reg _m[] =
	{
		{"start",        l_start},
		{"startBackground", l_startbg},
		{"share",        l_share},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}




#include "info.h"
extern "C"
{
SERVER_API int lib_register(lua_State* L);
SERVER_API int lib_version(lua_State* L);
SERVER_API const char* lib_name(lua_State* L);
SERVER_API int lib_main(lua_State* L);
}

SERVER_API int lib_register(lua_State* L)
{
	luaT_register<LuaServer>(L);
	return 0;
}

SERVER_API int lib_version(lua_State* L)
{
	return __revi;
}

SERVER_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Server";
#else
	return "Server-Debug";
#endif
}

SERVER_API int lib_main(lua_State* L)
{
	return 0;
}
