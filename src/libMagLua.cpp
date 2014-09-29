/******************************************************************************
* Copyright (C) 2008-2012 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#include "libMagLua.h"


#include "info.h"
#include "main.h"
#include "loader.h"
#include "import.h"
#include "modules.h"
#include "dofile.h"

#include <string.h>

#include "bootstrap.h"
#include "help.h"
#include "os_extensions.h"

#include <stdio.h>
#include <string>
#include <vector>
using namespace std;

static vector<string> args;


#include <errno.h>
#include <sys/time.h>
#include <sys/resource.h>

static int l_setrlimit(lua_State* L)
{
	if(!lua_isnumber(L, 1))
		return luaL_error(L, "setrlimit requires a resource");

	const int r = lua_tointeger(L, 1);

	if(!lua_istable(L, 2))
		return luaL_error(L, "setrlimit requires a table of 2 integers (soft and hard)");

	int sh[2];

	for(int i=0; i<2; i++)
	{
		lua_pushinteger(L, i+1);
		lua_gettable(L, 2);
		if(!lua_isnumber(L, -1))
			return luaL_error(L, "setrlimit requires a table of 2 integers (soft and hard)");

		sh[i] = lua_tointeger(L, -1);
		lua_pop(L, 1);
	}

	struct rlimit res;

	int err = getrlimit(r, &res);

	if(err == EINVAL)
	{
		return luaL_error(L, "resource is not valid");
	}

	res.rlim_cur = sh[0];
	res.rlim_max = sh[1];

	err = setrlimit(r, &res);

	if(err == EINVAL)
	{
		return luaL_error(L, "soft limit is larger than high limit");
	}

	if(err == EPERM)
	{
		return luaL_error(L, "An unprivileged process tried to use setrlimit() to increase a soft or hard limit above the current hard limit; the CAP_SYS_RESOURCE capability is required to do this.  Or, the process tried to use setrlimit() to increase the soft or hard RLIMIT_NOFILE limit above the current kernel maximum (NR_OPEN).");
	}


	return 0;
}

static int l_getrlimit(lua_State* L)
{
	if(!lua_isnumber(L, 1))
		return luaL_error(L, "getrlimit requires a resource");

	const int r = lua_tointeger(L, 1);

	struct rlimit res;

	int err = getrlimit(r, &res);

	if(err == EINVAL)
	{
		return luaL_error(L, "Resource is not valid");
	}

	lua_newtable(L);
	lua_pushinteger(L, 1);
	lua_pushinteger(L, res.rlim_cur);
	lua_settable(L, -3);

	lua_pushinteger(L, 2);
	lua_pushinteger(L, res.rlim_max);
	lua_settable(L, -3);

	return 1;
}

static void do_rlimit(lua_State* L)
{
	static const struct {
		const char* name;
		int value;
	} staticData[] = {
#ifdef RLIMIT_AS
		{"AS", RLIMIT_AS },
#endif
#ifdef RLIMIT_CORE
		{"CORE", RLIMIT_CORE },
#endif
#ifdef RLIMIT_CPU
		{"CPU", RLIMIT_CPU},
#endif
#ifdef RLIMIT_DATA
		{"DATA", RLIMIT_DATA},
#endif
#ifdef RLIMIT_FSIZE
		{"FSIZE", RLIMIT_FSIZE},
#endif
#ifdef RLIMIT_LOCKS
		{"LOCKS", RLIMIT_LOCKS},
#endif
#ifdef RLIMIT_MEMLOCK
		{"MEMLOCK", RLIMIT_MEMLOCK},
#endif
#ifdef RLIMIT_MSGQUEUE
		{"RLIMIT_MSGQUEUE", RLIMIT_MSGQUEUE},
#endif
		{ 0, 0 }
	};

	lua_newtable(L);
	
	int i=0; 
	while(staticData[i].name)
	{
		lua_pushstring(L, staticData[i].name);
		lua_pushinteger(L, staticData[i].value);
		lua_settable(L, -3);
		i++;
	}
	lua_setglobal(L, "RLIMIT");
	

	lua_pushcfunction(L, l_getrlimit);
	lua_setglobal(L, "getrlimit");
	lua_pushcfunction(L, l_setrlimit);
	lua_setglobal(L, "setrlimit");
}


static void lua_setupPreamble(lua_State* L, int sub_process)
{
	luaL_openlibs(L);

	register_os_extensions(L);
	
	if(luaL_dostringn(L, __dofile(), "=dofile.lua"))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
	}
		
	lua_getglobal(L, "dofile_add");
	lua_pushstring(L, "Help.lua");
	lua_pushstring(L, __help());
	lua_call(L, 2, 0);

	vector<string> _args = args;
	if(sub_process)
		_args.push_back("-q");
	lua_newtable(L);
	for(unsigned int i=0; i<_args.size(); i++)
	{
		lua_pushinteger(L, i+1);
		lua_pushstring(L, _args[i].c_str());
		lua_settable(L, -3);
	}
	lua_setglobal(L, "arg");

	if(sub_process)
	{
		lua_pushinteger(L, 1);
		lua_setglobal(L, "sub_process");
	}

#ifdef WIN32
	lua_pushstring(L, "WIN32");
#else
	lua_pushstring(L, "UNIX");
#endif
	lua_setglobal(L, "ENV");

	lua_pushcfunction(L, lua_getModulesInDirectory);
	lua_setglobal(L, "getModulesInDirectory");
	
	lua_pushcfunction(L, lua_loadfile);
	lua_setglobal(L, "loadModule");
	
	lua_pushcfunction(L, lua_unloadfile);
	lua_setglobal(L, "unloadModule");
	
	do_rlimit(L);

	lua_pushinteger(L, __revi);
	lua_setglobal(L, "__version"); //make version() in bootstrap
	
	lua_pushstring(L, __info);
	lua_setglobal(L, "__info"); //make info(x) in bootstrap
}	

int libMagLua(lua_State* L, int sub_process, int force_quiet)
{
	lua_setupPreamble(L, sub_process);

        if(luaL_dostringn(L, __bootstrap(), __bootstrap_name()))
        {
            fprintf(stderr, "%s\n", lua_tostring(L, -1));
            return 1; //luaL_error(L, lua_tostring(L, -1));
        }
	return 0;
}

int libMagLuaArgs(int argc, char** argv, lua_State* L, int sub_process, int force_quiet)
{
	for(int i=0; i<argc; i++)
		args.push_back(argv[i]);
	if(force_quiet)
		args.push_back("-q");
	
	return libMagLua(L, sub_process, force_quiet);
}
