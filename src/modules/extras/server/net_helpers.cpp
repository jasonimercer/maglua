#include "net_helpers.h"
#include "luamigrate.h"

#ifndef WIN32
#include <netinet/in.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#else
 #pragma warning(disable: 4251)
 #pragma warning(disable: 4996)
#include <stdio.h>
#include <WinSock.h>
typedef int socklen_t;
#define write(a,b,c) send(a,(const char*)b,c,0)
#define read(a,b,c) recv(a,(char*)b,c,0)
#endif
#include <errno.h>
#include <string.h>

using namespace std;


static int sock_valid(int sockfd)
{
	socklen_t optlen = 0;
	int rv = getsockopt(sockfd, SOL_SOCKET, SO_TYPE, NULL, &optlen);
#ifdef WIN32
	return 1;
#endif
	return rv >= 0;
	
	//return errno != EBADF;	   
}

#ifndef sure_write
#define sure_write(fd, data, sz, ok) sure_write_(fd, data, sz, ok, __FILE__, __LINE__)
#endif
int sure_write_(int fd, void* data, int sz, bool* ok, const char* file, int line)
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
#ifdef WIN32
			fprintf(stderr, "write(%i, %lX, %i) error: `%i' (%s:%i)\n", fd, (long)data, sz, WSAGetLastError(), file, line);
#else
			fprintf(stderr, "read(%i, %lX, %i) error: `%s' (%s:%i)\n", fd, (long)data, sz, strerror(errno), file, line);
#endif
			return msz;
		}
		msz += b; 
	}
	return msz;
}

#ifndef sure_read
#define sure_read(fd, data, sz, ok) sure_read_(fd, data, sz, ok, __FILE__, __LINE__)
#endif
int sure_read_(int fd, void* data, int sz, bool* ok, const char* file, int line)
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
#ifdef WIN32
			fprintf(stderr, "read(%i, %lX, %i) error: `%i' (%s:%i)\n", fd, (long)data, sz, WSAGetLastError(), file, line);
#else
			fprintf(stderr, "read(%i, %lX, %i) error: `%s' (%s:%i)\n", fd, (long)data, sz, strerror(errno), file, line);
#endif
			return msz;
		}
		msz += b; 
	}
	return msz;
}

#undef write
#undef read







LuaVariableGroup::LuaVariableGroup()
{
	
}

LuaVariableGroup::~LuaVariableGroup()
{
	clear();
}
	


//void LuaVariableGroup::add(char* data, int size)
//{
//	sizes.push_back(size);
//	variables.push_back(data);
//}

void LuaVariableGroup::clear()
{
	while(variables.size())
	{
		printf("clear %i\n", variables.size());
		free( variables.back() );
		printf("clear %i\n", variables.size());
		variables.pop_back();
		printf("clear %i\n", variables.size());
	}
	sizes.clear();
}

void LuaVariableGroup::write(int fd, bool& ok)
{
	unsigned int num = variables.size();
	ok = true;
	sure_write(fd, &num, sizeof(unsigned int), &ok);
	
	if(!ok) return;
	if(num)
	{
		sure_write(fd, &(sizes[0]), sizeof(int) * num, &ok);
		if(!ok) return;
	
		for(unsigned int i=0; i<num; i++)
		{
			sure_write(fd, variables[i], sizes[i], &ok);
			if(!ok) return;
		}
	}
}

void LuaVariableGroup::read(int fd, bool& ok)
{
	clear();
	
	unsigned int num = 0;
	sure_read(fd, &num, sizeof(unsigned int), &ok);
	
	if(!ok) return;
	if(num)
	{
		int* ss = new int[num];
		sure_read(fd, ss, sizeof(int) * num, &ok);
	
		for(unsigned int i=0; i<num; i++)
			sizes.push_back(ss[i]);
	
		delete [] ss;
		if(!ok) return;

	
		for(unsigned int i=0; i<num; i++)
		{
			char* b = (char*)malloc(sizes[i]+1); //can't mix new/malloc in WIN32, deeper things use malloc for buffer
			sure_read(fd, b, sizes[i], &ok);
			variables.push_back(b);
			if(!ok)
			{
				fprintf(stderr, "Failed to new_read (%s:%i)\n", __FILE__, __LINE__);
				return;
			}
		}
	}
}

void LuaVariableGroup::readState(lua_State* L)
{
	clear();

	int n = lua_gettop(L);
	if(n)
	{
		int sz;
		for(int i=0; i<n; i++)
		{
			variables.push_back(exportLuaVariable(L, i+1, &sz));
			sizes.push_back(sz);
		}
	}
}

void LuaVariableGroup::writeState(lua_State* L)
{
	for(unsigned int i=0; i<variables.size(); i++)
	{
		importLuaVariable(L, variables[i], sizes[i]);
	}
}



