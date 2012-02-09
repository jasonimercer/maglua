/******************************************************************************
* Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/


#ifdef _MPI
 #include <mpi.h>
#endif

#include "info.h"
#include "main.h"
#include "loader.h"
#include "import.h"
#include "modules.h"

#include "bootstrap.h"

#include <stdio.h>

int lua_setupPreamble(lua_State* L);

int main(int argc, char** argv)
{
	int suppress;
#ifdef _MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &suppress); //only rank 0 will chatter
#endif
	
	lua_State *L = lua_open();
	luaL_openlibs(L);
		
	lua_newtable(L);
	for(int i=0; i<argc; i++)
	{
		lua_pushinteger(L, i+1);
		lua_pushstring(L, argv[i]);
		lua_settable(L, -3);
	}
	lua_setglobal(L, "arg");
	
	lua_setupPreamble(L);

	if(luaL_dostring(L, __bootstrap))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
	}

	lua_close(L);
#ifdef _MPI
	MPI_Finalize();
#endif
	return 0;
}


int lua_setupPreamble(lua_State* L)
{
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
