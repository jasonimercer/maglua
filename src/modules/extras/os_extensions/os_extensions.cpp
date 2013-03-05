#include "os_extensions.h"

#include <unistd.h>
#include <sys/types.h>
#include <string.h>
#include <errno.h>
#include <sys/utsname.h>


#include "info.h"
extern "C"
{
OS_EXTENSIONS_API int lib_register(lua_State* L);
OS_EXTENSIONS_API int lib_version(lua_State* L);
OS_EXTENSIONS_API const char* lib_name(lua_State* L);
OS_EXTENSIONS_API int lib_main(lua_State* L);
}

static int l_getpid(lua_State* L)
{
	lua_pushinteger(L, getpid());
	return 1;
}

static int l_gethostname(lua_State* L)
{
	char n[256];
	gethostname(n, 256);
	lua_pushstring(L, n);
	return 1;
}

static int l_getdomainname(lua_State* L)
{
	char n[256];
	if(getdomainname(n, 256))
	{
		return luaL_error(L, strerror(errno));
	}
	lua_pushstring(L, n);
	return 1;
}

#ifndef _GNU_SOURCE
#define _GNU_SOURCE //for using the GNU CPU affinity
#endif
// (works with the appropriate kernel and glibc)
// Set affinity mask
#include <sched.h>
#include <stdio.h>
#include <unistd.h>

static int l_getcpuaffinity(lua_State* L)
{
	int NCPUs = sysconf(_SC_NPROCESSORS_CONF);
	cpu_set_t mask;
	
	if (sched_getaffinity(0, sizeof(mask), &mask) == -1)
	{
		return luaL_error(L, "sched_getaffinity(0, sizeof(mask), &mask)");
	}
	
	unsigned long lmask = *(unsigned long*)(&mask);
	
	lua_newtable(L);
	int j = 1;
	for(int i=0; i<NCPUs; i++)
	{
		if( (1<<i) & lmask)
		{
			lua_pushinteger(L, j); j++;
			lua_pushinteger(L, i+1);
			lua_settable(L, -3);
		}
	}
	
	
// 	printf("mask=%08X\n",  *(unsigned int*)(&mask));
	
	return 1;
}
    

static int l_getuname(lua_State* L)
{
	struct utsname u;

	if(uname(&u))
	{
		return luaL_error(L, strerror(errno));
	}
	
	lua_newtable(L);
	
	lua_pushstring(L, "sysname");
	lua_pushstring(L, u.sysname);
	lua_settable(L, -3);
	
	lua_pushstring(L, "nodename");
	lua_pushstring(L, u.nodename);
	lua_settable(L, -3);
	
	lua_pushstring(L, "release");
	lua_pushstring(L, u.release);
	lua_settable(L, -3);
		
	lua_pushstring(L, "version");
	lua_pushstring(L, u.version);
	lua_settable(L, -3);
	
	lua_pushstring(L, "machine");
	lua_pushstring(L, u.machine);
	lua_settable(L, -3);
	
#ifdef _GNU_SOURCE
	lua_pushstring(L, "domainname");
	lua_pushstring(L, u.domainname);
	lua_settable(L, -3);	
#endif

	return 1;
}

#include <iostream>
#include <fstream>
using namespace std;
// returns
//             RSS: The total amount of physical memory used by the task, in megabytes
//   Shared Memory: total shared memory (can include libraries), in megabytes
//  Private Memory: RSS - Shared Memory
static int l_getmemoryusage(lua_State* L)
{
    int tSize = 0, resident = 0, share = 0;
    ifstream buffer("/proc/self/statm");
    buffer >> tSize >> resident >> share;
    buffer.close();

    double page_size_mb = ((double)sysconf(_SC_PAGE_SIZE)) / (1024.0*1024.0); // in case x86-64 is configured to use 2MB pages

	double rss = resident * page_size_mb;
    double shared_mem = share * page_size_mb;

	
	lua_pushnumber(L, rss);
	lua_pushnumber(L, shared_mem);
	lua_pushnumber(L, rss - shared_mem);
	
	
    return 3;
}

OS_EXTENSIONS_API int lib_register(lua_State* L)
{
	lua_getglobal(L, "os");
	
	lua_pushstring(L, "hostname");
	lua_pushcfunction(L, l_gethostname);
	lua_settable(L, -3);
	
	lua_pushstring(L, "getCPUAffinity");
	lua_pushcfunction(L, l_getcpuaffinity);
	lua_settable(L, -3);
		
	lua_pushstring(L, "statm");
	lua_pushcfunction(L, l_getmemoryusage);
	lua_settable(L, -3);
	
	
	lua_pushstring(L, "domainname");
	lua_pushcfunction(L, l_getdomainname);
	lua_settable(L, -3);
			
	lua_pushstring(L, "uname");
	lua_pushcfunction(L, l_getuname);
	lua_settable(L, -3);
	
	lua_pushstring(L, "pid");
	lua_pushcfunction(L, l_getpid);
	lua_settable(L, -3);
	
	lua_setglobal(L, "os");
	
	return 0;
}

OS_EXTENSIONS_API int lib_version(lua_State* L)
{
	return __revi;
}

OS_EXTENSIONS_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "OS_Extensions";
#else
	return "OS_Extensions-Debug";
#endif
}

OS_EXTENSIONS_API int lib_main(lua_State* L)
{
	return 0;
}
