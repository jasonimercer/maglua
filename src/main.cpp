/******************************************************************************
* Copyright (C) 2008-2010 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#include "main.h"
#include "info.h"

int goodargs(int argc, char** argv)
{
	if(argc < 2)
	{
		cerr << __info << endl;
		cerr << "Please supply a Maglua script" << endl;
		return 0;
	}
	return 1;
}

void lua_addargs(lua_State* L, int argc, char** argv)
{
	lua_pushinteger(L, argc);
	lua_setglobal(L, "argc");

	lua_newtable(L);
	for(int i=0; i<argc; i++)
	{
		lua_pushinteger(L, i+1);
		lua_pushstring(L, argv[i]);
		lua_settable(L, -3);
	}
	lua_setglobal(L, "argv");
	
	lua_newtable(L);
	for(int i=2; i<argc; i++)
	{
		lua_pushinteger(L, i-1);
		lua_pushstring(L, argv[i]);
		lua_settable(L, -3);
	}
	lua_setglobal(L, "arg");
}

static int l_info(lua_State* L)
{
	string result;
	if(lua_gettop(L))
		result.append(lua_tostring(L, 1));

	for(int pos=0; __info[pos]; pos++)
		if(__info[pos] != '\n' || __info[pos+1] != 0)
		{
			result.append(1, __info[pos]);
			if(lua_gettop(L) && __info[pos] == '\n' && __info[pos+1])
					result.append(lua_tostring(L, 1));
		}

	lua_pushstring(L, result.c_str());
	return 1;
}

#ifdef _MPI
 #include <mpi.h>
#endif

int main(int argc, char** argv)
{
#ifdef _MPI
	MPI_Init(&argc, &argv);
#endif
	
	if(goodargs(argc, argv))
	{
		lua_State *L = lua_open();
		registerLibs(L);
		
		lua_pushcfunction(L, l_info);
		lua_setglobal(L, "info");

		lua_addargs(L, argc, argv);

		if(luaL_dofile(L, argv[1]))
		{
		  cerr << "Error:" << endl;
			cerr << lua_tostring(L, -1) << endl;
		}
		lua_close(L);
	}
	
#ifdef _MPI
	MPI_Finalize();
#endif
	
	return 0;
}
