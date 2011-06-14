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

#include <iostream>
#include <string>
#include <vector>
#include <string.h>
#include <errno.h>

using namespace std;

#include <dlfcn.h>
#include <dirent.h>
#include "main.h"
#include "luacommon.h"

// #include "libLuaClient.h"
// #include "libLuaSqlite.h"


vector<string> loaded;
vector<string> mod_dirs;

static int load_lib(int suppress, lua_State* L, const string& name)
{
	char buf[4096];
	
	for(unsigned int i=0; i<loaded.size(); i++)
	{
		if(name.compare(loaded[i]) == 0) //then already loaded
			return 1;
	}
	
	if(!suppress)
		cout << "  Loading Module: " << name << endl;
	
	void* handle = 0;
	
	
	for(unsigned int i=0; i<mod_dirs.size(); i++)
	{
		snprintf(buf, 4096, "%s/%s.so", mod_dirs[i].c_str(), name.c_str());

		handle = dlopen(buf, RTLD_LAZY);

		if(handle)
		{
		    // reset errors
			dlerror();
			break;
		}
	}

	if(!handle)
	{
		cerr << "Cannot load `" << name << "': " << dlerror() << '\n';
		return 1;
	}
	

	typedef int (*lua_func)(lua_State*);


	lua_func lib_deps = (lua_func) dlsym(handle, "lib_deps");
	const char *dlsym_error = dlerror();
	if(dlsym_error)
	{
        cerr << "Cannot load symbol `register_lib': " << dlsym_error << endl;
        dlclose(handle);
        return 1;
    }
    
	int n = lib_deps(L);
	vector<string> deps;
	for(int i=0; i<n; i++)
	{
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
	
    // reset errors
    dlerror();
	
    lua_func lib_register = (lua_func) dlsym(handle, "lib_register");

    if(!lib_register)
	{
        cerr << "Cannot load symbol lib_register': " << dlsym_error << endl;
        dlclose(handle);
        return 1;
    }
    

	lib_register(L);
// 	cout << "`" << name << "' loaded" << endl;
	loaded.push_back(name);
	
// 	dlclose(handle);
	
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

	
// 	load_lib(suppress, L, "LuaSqlite");
// 	load_lib(suppress, L, "LuaClient");
	
	registerSpinSystem(L);
	registerLLG(L);
	registerExchange(L);
	registerAppliedField(L);
	registerAnisotropy(L);
	registerDipole(L);
	registerRandom(L);
	registerThermal(L);
	registerInterpolatingFunction(L);
	registerInterpolatingFunction2D(L);
	registerDipoleDisordered(L);
// 	registerMagnetostatic(L);
	
	//registerSQLite(L);
	
    
    // close the library
//     cout << "Closing library...\n";
//     dlclose(handle);
	
	
//	registerLuaClient(L);
#ifdef _MPI
	registerMPI(L);
#endif
}
