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

// This is the main entry point for maglua. It loads in the 
// modules in the default directory and any others specified
// with -L options


extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#include "info.h"

#include <string.h>
#include <dlfcn.h>
#include <dirent.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <errno.h>
#ifdef _MPI
 #include <mpi.h>
#endif


using namespace std;


map<string,int> loaded;
vector<string> mod_dirs;

int registerLibs(int suppress, lua_State* L);
void lua_addargs(lua_State* L, int argc, char** argv);
static int l_info(lua_State* L);

int main(int argc, char** argv)
{
#ifdef _MPI
	MPI_Init(&argc, &argv);
#endif
	
	int suppress = 0;
	
	for(int i=0; i<argc; i++)
	{
		if(strcmp("-q", argv[i]) == 0)
			suppress = 1;
	}
	
	lua_State *L = lua_open();
	luaL_openlibs(L);

	int modbase = 1;
	lua_newtable(L);
#ifdef MAGLUAMODULESPATH
	lua_pushinteger(L, modbase);
	lua_pushstring(L, MAGLUAMODULESPATH);
	lua_settable(L, -3);
	modbase++;
#endif	

	for(int i=0; i<argc-1; i++)
	{
		if(strcmp("-L", argv[i]) == 0)
		{
			lua_pushinteger(L, modbase);
			lua_pushstring(L, argv[i+1]);
			lua_settable(L, -3);
			modbase++;
		}
	}

	lua_setglobal(L, "module_path");

	if(!suppress)
		cout << "MagLua r-" << __rev << " by Jason Mercer (c) 2011" << endl;
	
	if(!registerLibs(suppress, L))
	{
		int script = 0;

		lua_pushcfunction(L, l_info);
		lua_setglobal(L, "info");

		lua_addargs(L, argc, argv);
		
		// execute each lua script on the command line
		for(int i=1; i<argc; i++)
		{
			const char* fn = argv[i];
			int len = strlen(fn);
			
			if(len > 4)
			{
				if(strncasecmp(fn+len-4, ".lua", 4) == 0)
				{
					script++;
					if(luaL_dofile(L, fn))
					{
						cerr << "Error:" << endl;
						cerr << lua_tostring(L, -1) << endl;
					}
				}
			}
		}
		
		if(!script && !suppress)
		{
			cerr << "Please supply a Maglua script" << endl;
		}
		lua_close(L);
	
#ifdef _MPI
		MPI_Finalize();
#endif
	}
	
	return 0;
}



// add command line args to the lua state
// adding argc, argv and a table arg
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

// get info about the build
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



// Load a library into the process
//
// *****************************************************************
// all the folloing was broke. Idea was to load dependancies needed to
// load the library. Thing is you can't load the library to load the
// dependancies until the deps are loaded. What a dumb mistake...
// New method is to keep trying to load the libraries until they all
// loaded. If a round of loads doesn't produce any loads then something
// is really broke and we'll fail (this is good - don't want to keep
// retrying when there really is an error)
// *****************************************************************
static int load_lib(int suppress, lua_State* L, const string& name)
{
	char* buf = 0;
	int bufsize = 0;
	
	if(loaded[name] > 0)//then already loaded
	{
		return 3;
	}
	string src_dir;
	
	
	void* handle = 0;
	for(unsigned int i=0; i<mod_dirs.size(); i++)
	{
		int len = mod_dirs[i].length() + name.length() + 10;
		if(len > bufsize)
		{
			if(buf)
				delete [] buf;
			buf = new char[len];
			bufsize = len;
		}
		
		snprintf(buf, bufsize, "%s/%s.so", mod_dirs[i].c_str(), name.c_str());
		
		handle = dlopen(buf,  RTLD_NOW | RTLD_GLOBAL);

		if(handle)
		{
			src_dir = mod_dirs[i];
			break;
		}
	}

	if(!handle)
	{
		// don't report error, we may be able to deal with it
		return 2;
	}
	

	typedef int (*lua_func)(lua_State*);


	lua_func lib_register = (lua_func) dlsym(handle, "lib_register");

    if(!lib_register)
	{
		// we'll report this later 
		dlclose(handle);
		return 1;
    }
    
	lib_register(L); // call the register function from the library
	
	if(buf)
		delete [] buf;

	if(!suppress)
	{
		cout << "  Loading Module: " << name << endl;
		//cout << "    from " << src_dir << endl;
	}
	
	loaded[name]++;

	return 0;
}


int registerLibs(int suppress, lua_State* L)
{
	loaded.clear();
	mod_dirs.clear();

	map<string,int> unloaded;

	lua_getglobal(L, "module_path");
	lua_pushnil(L);
	while(lua_next(L, -2) != 0)
	{
		mod_dirs.push_back(lua_tostring(L, -1));
		lua_pop(L, 1);
	}
	lua_pop(L, 1); //pop table
	
	// make a list of all the modules we want to load
	for(unsigned int d=0; d<mod_dirs.size(); d++)
	{
		struct dirent *dp;

		DIR *dir = opendir(mod_dirs[d].c_str());
		if(dir)
		{
			while( (dp=readdir(dir)) != NULL)
			{
				const char* filename = dp->d_name;

				int len = strlen(filename);
				if(strncasecmp(filename+len-3, ".so", 3) == 0) //then ends with .so
				{
					char* n = new char[len+2];
					strncpy(n, filename, len-3);
					n[len-3] = 0;
					
					unloaded[n]++; //mark this module to be loaded
					
					delete [] n;
				}
			}
			closedir(dir);
		}
		else
		{
			cerr << "Failed to read directory `" << mod_dirs[d] << "': " << strerror(errno) << endl;
		}
		
	}
	
	
	map<string,int>::iterator mit;
	int num_loaded_this_round;
	
	do
	{
		num_loaded_this_round = 0;
		
		for(mit=unloaded.begin(); mit!=unloaded.end(); ++mit)
		{
			if( (*mit).second > 0 )
			{
				string name = (*mit).first;
				
				if(!load_lib(suppress, L, name))
				{
					(*mit).second = 0;
					num_loaded_this_round++;
				}

			}
		}
	}while(num_loaded_this_round > 0);
	
	
	// we've either loaded all the modules or there are some left with errors
	// check to see if and of the unloaded have finite value
	int bad_fail = 0;
	for(mit=unloaded.begin(); mit!=unloaded.end(); ++mit)
	{
		if( (*mit).second > 0)
		{
			// try to load it one more time(it will fail), we'll report any errors here
			int load_err = load_lib(suppress, L, (*mit).first);

			switch(load_err)
			{
				case 1:
					cerr << "Cannot load symbol `lib_register' in `" << (*mit).first << "': " << dlerror() << endl;
					break;

				case 2:
					cerr << "Cannot load `" << (*mit).first << "': " << dlerror() << endl;
					break;
				
				default:
					cerr << "Failed to load `" << (*mit).first << "'" << endl;
			}
			bad_fail++;
		}
	}
	
	return bad_fail;
}




