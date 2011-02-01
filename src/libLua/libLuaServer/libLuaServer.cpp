#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netdb.h>

#include <arpa/inet.h>

#include <string.h>
#include <stdlib.h>

#include "libLuaServer.h"

int LuaServer_::port;

vector<pthread_t*> LuaServer_::commThread;

sem_t LuaServer_::luaThreadSem;
pthread_t LuaServer_::rootThread;
int LuaServer_::nextThreadID = 0;
vector<LuaThread*> LuaServer_::runningLuathreads;
luafunc LuaServer_::registerCallback = 0;

sem_t LuaServer_::addCommSem;
sem_t LuaServer_::updateSem;

vector<LuaComm*> LuaServer_::comms;

#define SHUTDOWN        0
#define UPLOADLUA       10
#define REMOTELUA       11




#ifndef __sure_write
#define __sure_write(fd, data, sz) __sure_write_(fd, data, sz, __FILE__, __LINE__)
#endif
int __sure_write_(int fd, void* data, int sz, const char* file, int line)
{
	int b;
	int msz = write(fd, data, sz);
	while(msz < sz)
	{
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

#ifndef __sure_read
#define __sure_read(fd, data, sz) __sure_read_(fd, data, sz, __FILE__, __LINE__)
#endif
int __sure_read_(int fd, void* data, int sz, const char* file, int line)
{
// 	printf("Reading %i from (%s:%i)\n", sz, file, line);
	int b;
	int msz = read(fd, data, sz);
	while(msz < sz)
	{
		b = read(fd, &((char*)data)[msz], sz-msz);
		if(b == -1)
		{
			fprintf(stderr, "read(%i, %lX, %i) error: `%s' (%s:%i)\n", fd, (long)data, sz, strerror(errno), file, line);
			return msz;
		}
		msz += b; 
	}

	return msz;
}

void __threadSignalHandler(int signo);


void* __threadCommMain(void* args)
{
	pair<int,char*>* p = (pair<int,char*>*)args;

	int   fd = p->first;
	char* cn = p->second;

	LuaComm* mc = new LuaComm(cn);

	LuaServer.addComm(mc);

	mc->do_comm(fd);

	LuaServer.removeComm(mc);
	printf("thread done\n");
}


bool LuaServer_::init(int argc, char** argv)
{
// 	if(argc == 1 || argc > 3)
// 	{
// 		cerr << "Expected: " << argv[0] << " tagfile [port number]" << endl;
// 		return false;
// 	}

	rootThread = pthread_self();

	signal(SIGUSR1, __threadSignalHandler); //install sig handler

	if(argc == 2) 
		port = atoi(argv[1]);
	else
		port = 55000;

	sem_init(&addCommSem,   0, 1);
	sem_init(&updateSem,    0, 1);
	sem_init(&luaThreadSem, 0, 1);

	return true;
}

void LuaServer_::addError(string e, LuaComm* c)
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
void LuaServer_::addInfo(string e, LuaComm* c)
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


void LuaServer_::serve()
{
	int t, s;
	if((s = establish(port)) < 0)
 	{
		perror("establish");
		exit(1);
	}

	char connectionName[512];
	for(;;)
	{
		if((t= get_connection(s, connectionName)) < 0)
		{
			fprintf(stderr, "ERROR: %s\n", strerror(errno));
			if(errno == EINTR) /* EINTR might happen on accept(), */
				continue; /* try again */
			perror("accept"); /* bad */
				exit(1);
		}

		pthread_t* thread = new pthread_t();

		pair<int,char*> p(t, connectionName);

		commThread.push_back(thread);

		pthread_create(thread, NULL, __threadCommMain, (void *)&p);
	}
}

int LuaServer_::establish(unsigned short portnum)
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
		return(-1);

	sa.sin_family= hp->h_addrtype; /* this is our host address */
	sa.sin_port= htons(portnum); /* this is our port number */

	if((s= socket(AF_INET, SOCK_STREAM, 0)) < 0) /* create socket */
		return(-1);

	if(bind(s,(struct sockaddr *)&sa,sizeof(struct sockaddr_in)) < 0) 
	{
		close(s);
		perror("bind");
		return(-1); /* bind address to socket */
	}

	listen(s, 3); /* max # of queued connects */
	return(s);
}

int LuaServer_::get_connection(int s, char* connectionName)
{
	sockaddr_in cli_addr;
	socklen_t addrlen = 1024;

	int t;
	printf("waiting for connection\n");
	if ((t = accept(s, (struct sockaddr*)(&cli_addr), &addrlen)) < 0)
	{
		fprintf(stderr, "ERROR: %s\n", strerror(errno));
		return(-1);
	}

	printf("new connection from %s:%i\n", inet_ntoa( cli_addr.sin_addr), cli_addr.sin_port);
	sprintf(connectionName, "%s:%i", inet_ntoa( cli_addr.sin_addr), cli_addr.sin_port);


	return(t);
}

void LuaServer_::addComm(LuaComm* comm)
{
	sem_wait(&addCommSem);
		comms.push_back(comm);	
	sem_post(&addCommSem);
}


void LuaServer_::removeComm(LuaComm* comm)
{
	sem_wait(&addCommSem);

	vector<LuaComm*>::iterator it;

	for(it = comms.begin();it != comms.end(); ++it)
	{
		if( (*it) == comm )
			break;
	}

	if((*it) == comm)
		comms.erase(it);

	sem_post(&addCommSem);
}

void threadShutdown()
{
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
}

void __threadSignalHandler(int signo)
{
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
}

LuaThread* LuaServer_::getThread(int pid)
{
	LuaThread* t = 0;
	sem_wait(&luaThreadSem);
		vector<LuaThread*>::iterator it;
	
		for(it=LuaServer.runningLuathreads.begin(); it != LuaServer.runningLuathreads.end(); ++it)
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
	signal(SIGUSR1, __threadSignalHandler); //install sig handler

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
			LuaServer.addError(s, c); //only error back to src client, not all clients
		}
	}

	sem_wait(&LuaServer.luaThreadSem);
	vector<LuaThread*>::iterator it;

	for(it=LuaServer.runningLuathreads.begin(); it != LuaServer.runningLuathreads.end(); ++it)
	{
		if(pthread_equal( (*it)->thread, pthread_self() ))
		{
			LuaThread* lv = *it;
			LuaServer.runningLuathreads.erase(it);

			lua_close(lv->L);
			delete lv;

			break;
		}
	}
	sem_post(&LuaServer.luaThreadSem);
}


int LuaServer_::addLuaFunction(lua_Variable* vars, int nvars, LuaComm* comm)
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

	pthread_create(&(thread->thread), NULL, __threadLuaMain, (void *)thread);
	pthread_detach(thread->thread); //we wont be joining
	return retval;
}

lua_Variable* LuaServer_::executeLuaFunction(lua_Variable* vars, int nvars, LuaComm* comm, int* nret)
{
	lua_State* L;
	L = lua_open();
	luaL_openlibs(L);
	if(registerCallback)
		registerCallback(L);

	for(int i=0; i<nvars; i++)
		importLuaVariable(L, &vars[i]);

	int nargs = lua_gettop(L)-1;

	if(lua_pcall(L, nargs, LUA_MULTRET, 0))
	{
		string s = lua_tostring(L, -1);		
		cerr << s << endl;
		LuaServer.addError(s, comm); //only error back to src client, not all clients
	}

	//we started with a clean stack so whatever's on it now gets pushed back to the client
	lua_Variable* retvar = 0;

	int n = lua_gettop(L);
	*nret = n;

	if(n)
		retvar = (lua_Variable*)malloc(sizeof(lua_Variable)*n);

	for(int i=0; i<n; i++)
	{
		initLuaVariable(&retvar[i]);
		exportLuaVariable(L, i+1, &retvar[i]);
	}

	lua_close(L);

	return retvar;
}




LuaComm::LuaComm(const char* Name)
{
	sem_init(&qSem, 0, 1);
	strncpy(_name, Name, 512);

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
	int loop = 1;
	int cmd;

	int i, j, k, b;
	int yesno;

	
	int numluavars;
	lua_Variable* luavars;
	lua_Variable* lvar;

	int numReturnLuavars; 
	lua_Variable* returnlvars;

	int intpair[2];

	char* chardata;
	void* data;

	string ss;

	while(loop)
	{
		b = 0;

		b += __sure_read(fd, &cmd, sizeof(int));
		b += __sure_read(fd, &j,   sizeof(int));

		if(b < 2*sizeof(int))
			cmd = SHUTDOWN;

		fflush(stdout);
	
		yesno = 0;
		switch(cmd)
		{
			case SHUTDOWN:
				loop = 0;
			break;
		

			case UPLOADLUA:
				//lua_Variable* luavars;
				numluavars = j;
				luavars = (lua_Variable*)malloc(sizeof(lua_Variable) * numluavars);

				for(i=0; i<numluavars; i++)
					rawVarRead(fd, &luavars[i]);
				
				//add it to a state
				LuaServer.addLuaFunction(luavars, numluavars, this);

				for(i=0; i<numluavars; i++)
					freeLuaVariable(&luavars[i]);

				free(luavars);
			break;

			case REMOTELUA:
				numluavars = j;
				luavars = (lua_Variable*)malloc(sizeof(lua_Variable) * numluavars);

				for(i=0; i<numluavars; i++)
					rawVarRead(fd, &luavars[i]);
				
				//add it to a state
				returnlvars = LuaServer.executeLuaFunction(luavars, numluavars, this, &numReturnLuavars);

				__sure_write(fd, &numReturnLuavars, sizeof(int));

				if(numReturnLuavars)
				{
					for(i=0; i<numReturnLuavars; i++)
					{
						rawVarWrite(fd, &returnlvars[i]);
						freeLuaVariable(&returnlvars[i]);
					}
					free(returnlvars);
				}

				for(i=0; i<numluavars; i++)
					freeLuaVariable(&luavars[i]);
				free(luavars);
			break;

		}
	}


	cout << "Client disconnect" << endl;
}


void LuaComm::rawVarRead(int fd, lua_Variable* v)
{
	__sure_read(fd, v, sizeof(lua_Variable));

	if(v->ssize)
	{
		v->s = (char*)malloc(sizeof(char)*v->ssize);
		__sure_read(fd, v->s, sizeof(char)*v->ssize);
	}

	if(v->chunksize)
	{
		v->funcchunk = (char*)malloc(sizeof(char)*v->chunksize);
		__sure_read(fd, v->funcchunk, v->chunksize); 
	}

	if(v->listlength)
	{
		v->listKey = (lua_Variable*)malloc(sizeof(lua_Variable)*v->listlength);
		v->listVal = (lua_Variable*)malloc(sizeof(lua_Variable)*v->listlength);
		for(int i=0; i<v->listlength; i++)
		{
			rawVarRead(fd, &v->listKey[i]);
			rawVarRead(fd, &v->listVal[i]);
		}
	}
}

void LuaComm::rawVarWrite(int fd, lua_Variable* v)
{
	__sure_write(fd, v, sizeof(lua_Variable));

	if(v->ssize)
		__sure_write(fd, v->s, sizeof(char)*v->ssize);

	if(v->chunksize)
		__sure_write(fd, v->funcchunk, v->chunksize); 

	if(v->listlength)
	{
		for(int i=0; i<v->listlength; i++)
		{
// 			printf("sending key %i of %i\n", i+1, v->listlength); 
			rawVarWrite(fd, &v->listKey[i]);
// 			printf("sending val %i of %i\n", i+1, v->listlength); 
			rawVarWrite(fd, &v->listVal[i]);
		}
	}
}


