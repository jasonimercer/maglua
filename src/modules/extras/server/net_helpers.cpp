#include "net_helpers.h"
#include "luamigrate.h"

#include <netinet/in.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <errno.h>
#include <string.h>

using namespace std;


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
			fprintf(stderr, "read(%i, %lX, %i) error: `%s' (%s:%i)\n", fd, (long)data, sz, strerror(errno), file, line);
			return msz;
		}
		msz += b; 
	}
	return msz;
}









LuaVariableGroup::LuaVariableGroup()
{
	
}

LuaVariableGroup::~LuaVariableGroup()
{
	clear();
}
	


void LuaVariableGroup::add(char* data, int size)
{
	sizes.push_back(size);
	variables.push_back(data);
}

void LuaVariableGroup::clear()
{
	while(variables.size())
	{
		delete [] variables.back();
		variables.pop_back();
	}
	sizes.clear();
}

void LuaVariableGroup::write(int fd, bool& ok)
{
	unsigned int num = variables.size();
	ok = true;
	sure_write(fd, &num, sizeof(unsigned int), &ok);
	
	if(!ok) return;
	
	sure_write(fd, &(sizes[0]), sizeof(int) * num, &ok);
	if(!ok) return;
	
	for(unsigned int i=0; i<num; i++)
	{
		sure_write(fd, variables[i], sizes[i], &ok);
		if(!ok) return;
	}
}

void LuaVariableGroup::read(int fd, bool& ok)
{
	clear();
	
	unsigned int num = 0;
	sure_read(fd, &num, sizeof(unsigned int), &ok);
	
	if(!ok) return;
	
	int* ss = new int[num];
	sure_read(fd, ss, sizeof(int) * num, &ok);
	
	for(int i=0; i<num; i++)
		sizes.push_back(ss[i]);
	
	delete [] ss;
	if(!ok) return;

	
	for(unsigned int i=0; i<num; i++)
	{
		char* b = new char[sizes[i]+1];
		sure_read(fd, b, sizes[i], &ok);
		variables.push_back(b);
		if(!ok) return;
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



