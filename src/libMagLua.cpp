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

#include "bootstrap.h"

#include <stdio.h>
#include <string>
#include <vector>
using namespace std;

static vector<string> args;


static void lua_setupPreamble(lua_State* L, int sub_process)
{
	luaL_openlibs(L);
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

void libMagLua(lua_State* L, int sub_process, int force_quiet)
{
	lua_setupPreamble(L, sub_process);
	
	if(luaL_dostring(L, __bootstrap))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
	}
}

void libMagLuaArgs(int argc, char** argv, lua_State* L, int sub_process, int force_quiet)
{
	for(int i=0; i<argc; i++)
		args.push_back(argv[i]);
	if(force_quiet)
		args.push_back("-q");
	
	libMagLua(L, sub_process, force_quiet);
}
