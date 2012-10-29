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


static void lua_setupPreamble(lua_State* L, int sub_process)
{
	luaL_openlibs(L);

	register_os_extensions(L);
	
	luaL_dostring(L, __dofile());
		
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
	lua_setglobal(L, "getModuleDirectory");
	
	lua_pushcfunction(L, lua_loadfile);
	lua_setglobal(L, "loadModule");
	
	lua_pushinteger(L, __revi);
	lua_setglobal(L, "__version"); //make version() in bootstrap
	
	lua_pushstring(L, __info);
	lua_setglobal(L, "__info"); //make info(x) in bootstrap
}

static int pushtraceback(lua_State* L)
{
	lua_getglobal(L, "debug");
	if (!lua_istable(L, -1)) {
		lua_pop(L, 1);
		return 1;
	}
	lua_getfield(L, -1, "traceback");
	if (!lua_isfunction(L, -1)) {
		lua_pop(L, 2);
		return 1;
	}
	lua_remove(L, 1); //remove debug table
	return 0;
}

static void trim_err(const char* e, char* b)
{
	int n = strlen(e)+1;
	if(n > 2048)
		n = 2048;
	
	memcpy(b, e, n);
	b[2048] = 0;
	for(int i=n-1; i>0; i--)
	{
		if(strncmp(b+i, "\t[C]: in function 'dofile'", 26) == 0)
		{
			b[i] = 0;
			if(i && b[i-1] == '\n')
				b[i-1] = 0;
			return;
		}
	}
	
}
	

int libMagLua(lua_State* L, int sub_process, int force_quiet)
{
	int ret = 0;
	lua_setupPreamble(L, sub_process);
	
	pushtraceback(L);
	
	if(luaL_loadstring(L, __bootstrap()))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return 2;
	}
	
	if(lua_pcall(L, 0, 0, -2))
	{
		const char* err = lua_tostring(L, -1);
		lua_getglobal(L, "debug");
		lua_getfield(L, -1, "trim_error");
		lua_pushstring(L, err);

		if(lua_pcall(L, 1, 1, 0))
		{
			fprintf(stderr, "Error in error handler: %s\n", lua_tostring(L, -1));
			fprintf(stderr, "Original error: %s\n", err);
		}
		else
		{
			fprintf(stderr, "%s\n", lua_tostring(L, -1));
		}
		ret = 1;
	}
	return ret;
}

int libMagLuaArgs(int argc, char** argv, lua_State* L, int sub_process, int force_quiet)
{
	for(int i=0; i<argc; i++)
		args.push_back(argv[i]);
	if(force_quiet)
		args.push_back("-q");
	
	return libMagLua(L, sub_process, force_quiet);
}
