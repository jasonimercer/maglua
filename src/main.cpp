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
#include <errno.h>
#ifdef _MPI
 #include <mpi.h>
#endif


using namespace std;


vector<string> loaded;
vector<string> mod_dirs;

void registerLibs(int suppress, lua_State* L);
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
	
	registerLibs(suppress, L);
	bool script = false;

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
				script = true;
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



// load a library into the process
static int load_lib(int suppress, lua_State* L, const string& name)
{
	char buf[4096];
	
	for(unsigned int i=0; i<loaded.size(); i++)
	{
		if(name.compare(loaded[i]) == 0) //then already loaded
			return 1;
	}
	
	
	void* handle = 0;
	
	
	for(unsigned int i=0; i<mod_dirs.size(); i++)
	{
		snprintf(buf, 4096, "%s/%s.so", mod_dirs[i].c_str(), name.c_str());
		
		// loading lazy because we may need deps first
		// will load full at end of function
		handle = dlopen(buf, RTLD_LAZY);

		if(handle)
		{
			dlerror(); // reset errors
			break;
		}
	}

	if(!handle)
	{
		cerr << " Cannot load `" << name << "': " << dlerror() << '\n';
		return 1;
	}
	

	typedef int (*lua_func)(lua_State*);

#if 0
	lua_func lib_name = (lua_func) dlsym(handle, "lib_name");
	
	const char* dlsym_error_name = dlerror();
	if(dlsym_error_name)
	{
//         cerr << "Cannot load symbol `lib_name': " << dlsym_error_name << endl;
//         dlclose(handle);
//         return 1;
    }
    else
	{
		int n = lib_name(L);
		printf("n(name) = %i\n", n);
		while(n)
		{
			printf("%s\n", lua_tostring(L, -1));
			n--;
		}
	}
#endif

	lua_func lib_deps = (lua_func) dlsym(handle, "lib_deps");
	const char *dlsym_error = dlerror();
	if(dlsym_error)
	{
        cerr << "Cannot load symbol lib_deps': " << dlsym_error << endl;
        dlclose(handle);
        return 1;
    }
	int n = lib_deps(L);

	vector<string> deps;
	for(int i=0; i<n; i++)
	{
		cout << lua_tostring(L, -1) << endl;
		deps.push_back(lua_tostring(L, -1));
		lua_pop(L, 1);
	}
	
	// for each dependancy, check the loaded list
	// if it does not exist, load it.
	for(int i=0; i<n; i++)
	{
		bool dep_loaded = false;
		for(unsigned int j=0; j<loaded.size(); j++)
		{
			if(deps[i].compare(loaded[j]) == 0)
				dep_loaded = true;
		}
		
		cout << name << " depends on " << deps[i] << endl;
		if(!dep_loaded)
		{
			if(load_lib(suppress, L, deps[i]))
			{
				cerr << "Dependancy for `" << name << "' failed to load" << endl;
				return 1;
			}
		}
	}
	
	dlerror();    // reset errors

	lua_func lib_register = (lua_func) dlsym(handle, "lib_register");

    if(!lib_register)
	{
		cerr << "Cannot load symbol lib_register': " << dlsym_error << endl;
		dlclose(handle);
		return 1;
    }
    

	lib_register(L); // call the register function from the library
	loaded.push_back(name);
	
	if(!suppress)
		cout << "  Loading Module: " << name << endl;
	dlclose(handle);
	handle = dlopen(buf, RTLD_NOW | RTLD_GLOBAL);

	return 0;
}


void registerLibs(int suppress, lua_State* L)
{
	loaded.clear();
	mod_dirs.clear();

	lua_getglobal(L, "module_path");
	lua_pushnil(L);
	while(lua_next(L, -2) != 0)
	{
		mod_dirs.push_back(lua_tostring(L, -1));
		lua_pop(L, 1);
	}
	lua_pop(L, 1); //pop table
	
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
					
					load_lib(suppress, L, n);

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
}




