#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netdb.h>
#include <string>
#include <arpa/inet.h>

using namespace std;

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

sem_t LuaServer_::availableStates;


vector<LuaComm*> LuaServer_::comms;

#define SHUTDOWN        0
#define UPLOADLUA       10
#define REMOTELUA       11

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


struct fd_name
{
	int fd;
	char* name;
};

void* __threadCommMain(void* args)
{
	fd_name* a = (fd_name*)args;

	int   fd = a->fd;
	const char* cn = a->name;

	LuaComm* mc = new LuaComm(cn);

	LuaServer.addComm(mc);

	mc->do_comm(fd);

// 	cout << "*****  SHUTTING DOWN " << fd << endl;
	shutdown(fd, SHUT_RDWR);
	close(fd);
		
	LuaServer.removeComm(mc);
	//printf("thread done\n");
	
	free(a->name);
	free(a);
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
	
	sem_init(&availableStates,  0, 10);

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
	int fd, s;
	if((s = establish(port)) < 0)
 	{
		perror("establish");
		exit(1);
	}

	char connectionName[512];
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

		fd_name* a = (fd_name*)malloc(sizeof(fd_name));
		a->fd = fd;
		a->name = (char*)malloc(strlen(connectionName)+1);
		strcpy(a->name, connectionName);
		
		commThread.push_back(thread);

		pthread_create(thread, NULL, __threadCommMain, (void *)a);
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

	listen(s, 128); /* max # of queued connects */
	return(s);
}

int LuaServer_::get_connection(int s, char* connectionName)
{
	sockaddr_in cli_addr;
	socklen_t addrlen = 1024;

	int fd;
	printf("waiting for connection\n");
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
	//limit the number of simultaneous lua states running
	//this can cause a problem if the server runs code that
	//makes a new connection to itself and that connection cannot run
	sem_wait(&availableStates);
	
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
	{
		retvar = (lua_Variable*)malloc(sizeof(lua_Variable)*n);

		for(int i=0; i<n; i++)
		{
			initLuaVariable(&retvar[i]);
			exportLuaVariable(L, i+1, &retvar[i]);
		}
	}
	
	lua_close(L);

	sem_post(&availableStates);

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
// 	cout << "DO COMM " << fd << endl;
	int loop = 1;
	int ok = 1;
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

	while(loop && ok)
	{
		b = 0;

		b += sure_read(fd, &cmd, sizeof(int), &ok); if(!ok) continue; //and exit loop
		b += sure_read(fd, &j,   sizeof(int), &ok); if(!ok) continue; //and exit loop

		if(b < 2*sizeof(int))
			cmd = SHUTDOWN;

		fflush(stdout);
	
		yesno = 0;
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
// 				#define CFD "(" << fd << ")"
// 				cout << "REMOTE START " << CFD << endl;
				numluavars = j;
				luavars = (lua_Variable*)malloc(sizeof(lua_Variable) * (numluavars+1)); // +1 so it's not 0

				for(i=0; i<numluavars; i++)
				{
					if(!rawVarRead(fd, &luavars[i]))
					{
						for(int k=0; i<i-1; k++)//-1?
						{
							freeLuaVariable(&luavars[k]);
						}
						free(luavars);
						ok = 0; continue;
					}
				}
				
				//add it to a state
// 				cout << "EX START"<< CFD  << endl;
				returnlvars = LuaServer.executeLuaFunction(luavars, numluavars, this, &numReturnLuavars);
// 				cout << "EX END"<< CFD  << endl;
				
				for(i=0; i<numluavars; i++)
					freeLuaVariable(&luavars[i]);
				free(luavars);
				
				
// 				cout << "RETURN START (" << numReturnLuavars << ")"<< CFD  << endl;
				sure_write(fd, &numReturnLuavars, sizeof(int), &ok);
				if(!ok) continue;

				if(numReturnLuavars)
				{
					for(i=0; i<numReturnLuavars; i++)
					{
						if(!rawVarWrite(fd, &returnlvars[i]))
							ok = false;
						freeLuaVariable(&returnlvars[i]);
					}
				}
// 				cout << "RETURN END"<< CFD  << endl;
				
				if(returnlvars)
					free(returnlvars);
// 				cout << "REMOTE END"<< CFD  << endl;
				
			break;
		}
	}

	cout << "Client disconnect" << endl;
}


bool LuaComm::rawVarRead(int fd, lua_Variable* v)
{
	int ok = 1;
	
// 	cout << fctrl(fd, F_GETFD);
	initLuaVariable(v);
	int v3[3];
	
	sure_read(fd, v3, sizeof(int)*3, &ok);
	if(!ok) return false;
	
	v->chunksize = v3[0];
	v->chunklength = v3[0];
	v->listlength = v3[1];
	v->type = v3[2];
	
	if(v->chunksize)
	{
		v->chunk = (char*)malloc(sizeof(char)*v->chunklength);
		sure_read(fd, v->chunk, v->chunklength, &ok); 
		if(!ok) return false;
	}

	if(v->listlength)
	{
		v->listKey = (lua_Variable*)malloc(sizeof(lua_Variable)*v->listlength);
		v->listVal = (lua_Variable*)malloc(sizeof(lua_Variable)*v->listlength);
		for(int i=0; i<v->listlength; i++)
		{
			if(!rawVarRead(fd, &v->listKey[i])) return false;
			if(!rawVarRead(fd, &v->listVal[i])) return false;
		}
	}
	return true;
}

bool LuaComm::rawVarWrite(int fd, lua_Variable* v)
{
	int ok = 1;
	int v3[3];
	v3[0] = v->chunklength;
	v3[1] = v->listlength;
	v3[2] = v->type;
	
	sure_write(fd, v3, sizeof(int)*3, &ok); if(!ok) return false;

	if(v->chunklength)
	{
		sure_write(fd, v->chunk, v->chunklength, &ok); 
		if(!ok) return false;
	}

	if(v->listlength)
	{
		for(int i=0; i<v->listlength; i++)
		{
			if(!rawVarWrite(fd, &v->listKey[i])) return false;
			if(!rawVarWrite(fd, &v->listVal[i])) return false;
		}
	}
	return true;
}
