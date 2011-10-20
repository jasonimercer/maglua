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

#ifndef WIN32
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <dirent.h>
#define HOME "HOME"
#define SO_EXT "so"
#define MAGLUA_SETUP_PATH "/.maglua.d/module_path.lua"
#define _handle void*
#else
 #include <windows.h>
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #define snprintf _snprintf
 #define HOME "APPDATA"
 #define SO_EXT "dll"
 #define MAGLUA_SETUP_PATH "\\maglua\\module_path.lua"
 #define _handle HINSTANCE
#endif

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <errno.h>
#ifdef _MPI
 #include <mpi.h>
#endif
#include "setup_code.h"

using namespace std;


map<string,int> loaded;
vector<string> mod_dirs;

int getmoddirs(vector<string>& mds, const int argc, char** argv);
int registerLibs(int suppress, lua_State* L);
void lua_addargs(lua_State* L, int argc, char** argv);
static int l_info(lua_State* L);
void print_help();
const char* reference();
// 
// command line switches:
//  -L                  add moduel dir to search path
//  -q                  run quietly, omit some startup info printing
//  --module_path       print primary mod dir
//  --setup mod_dir     setup startup files in $(HOME)/.maglua.d
//                        with mod_dir in the list of module paths  
//  -h --help           show this help
// 

int main(int argc, char** argv)
{
	int suppress = 0;
	int shutdown = 0;
#ifdef _MPI
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &suppress); //only rank 0 will chatter
#endif
	if(!suppress) //prevent slaves from getting legal. 
		fprintf(stderr, "This evaluation version of MagLua is for private, non-commercial use only\n");

	for(int i=0; i<argc; i++)
	{
		if(strcmp("-q", argv[i]) == 0)
			suppress = 1;

		if(strcmp("-h", argv[i]) == 0 || strcmp("--help", argv[i]) == 0)
		{
			print_help();
			shutdown = 1;
		}
		
		if(strcmp("--module_path", argv[i]) == 0)
		{
			vector<string> mp;
			getmoddirs(mp, argc, argv);
			for(unsigned int i=0; i<mp.size() && i < 1; i++)
			{
				cout << mp[i] << endl;
			}
			shutdown = 1;
		}

		if(strcmp("--setup", argv[i]) == 0)
		{
			if(i < argc-1)
			{
				lua_State *L = lua_open();
				luaL_openlibs(L);

				if(luaL_dostring(L, setup_lua_code))
				{
					fprintf(stderr, "%s\n", lua_tostring(L, -1));
				}
				
				printf("\n\n%s\n\n", setup_lua_code);
				
				lua_getglobal(L, "setup");
				lua_pushstring(L, argv[i+1]);
				
				if(lua_pcall(L, 1,0,0))
				{
					fprintf(stderr, "%s\n", lua_tostring(L, -1));
				}
				
				lua_close(L);
				shutdown = 1;
			}
			
		}
	}
	
	if(shutdown)
	{
		#ifdef _MPI
			MPI_Finalize();
		#endif
		return 0;
	}
	
#ifdef _MPI
	{
		//suppress startup messages for non-root rank
		int rank;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		suppress |= rank;
	}
#endif

	
	lua_State *L = lua_open();
	luaL_openlibs(L);

	int modbase = 1;
	lua_newtable(L);
	
	
	{
		vector<string> m;
		getmoddirs(m, argc, argv);
		for(unsigned int i=0; i<m.size(); i++)
		{
			lua_pushinteger(L, modbase);
			lua_pushstring(L, m[i].c_str());
			lua_settable(L, -3);
			modbase++;
		}
	

	}

	lua_setglobal(L, "module_path");

	if(!suppress)
	{
		cout << "MagLua r-" << __rev << " by Jason Mercer (c) 2011" << endl;
		cout << endl;
		cout << reference() << endl;
		cout << endl;
	}
	
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
		
		if(!script)
		{
			cerr << "Please supply a MagLua script (*.lua)" << endl;
		}
		lua_close(L);
	
#ifdef _MPI
		MPI_Finalize();
#endif
	}
	
	return 0;
}


int getmoddirs(vector<string>& mds, const int argc, char** argv)
{
	for(int i=0; i<argc-1; i++)
	{
		if(strcmp("-L", argv[i]) == 0)
		{
			mds.push_back(argv[i+1]);
			i++;
		}
	}


	lua_State *L = lua_open();
	luaL_openlibs(L);
	
	int home_len = 0;
#ifndef WIN32
	char* home = getenv(HOME);
#else
	char* home;
	size_t foo;
	_dupenv_s(&home, &foo, HOME);
#endif
	if(home)
	{
		home_len = strlen(home);
	}
	
	char* init_file = new char[home_len + 128];
	
	init_file[0] = 0;
	if(home)
	{
#ifndef WIN32
		strcpy(init_file, home);
#else
		strcpy_s(init_file, home_len+128, home);
#endif
	}
	else
	{
#ifndef WIN32
		strcpy(init_file, ".");
#else
		strcpy_s(init_file, home_len+128, ".");
#endif
	}
	
#ifndef WIN32
	strcat(init_file, MAGLUA_SETUP_PATH);
#else
	strcat_s(init_file, home_len+128, MAGLUA_SETUP_PATH);
#endif

	if(!luaL_dofile(L, init_file))
	{
		lua_getglobal(L, "module_path");
		if(lua_istable(L, -1))
		{
			lua_pushnil(L);
			while(lua_next(L, -2))
			{
				if(lua_isstring(L, -1))
				{
					mds.push_back(lua_tostring(L, -1));
				}
				lua_pop(L, 1);
			}
		}
		else
		{
			if(lua_isstring(L, -1))
			{
				mds.push_back(lua_tostring(L, -1));
			}
		}
	}
	else
	{
		printf("%s\n", lua_tostring(L, -1));
	}
	
	delete [] init_file;

	
#ifdef WIN32
	free(home);
#endif
	lua_close(L);
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

	if(lua_gettop(L))
		result.append(lua_tostring(L, 1));

	result.append("\nModules: ");

	map<string,int>::iterator mit;
	for(mit=loaded.begin(); mit!=loaded.end(); ++mit)
	{
		result.append((*mit).first);
		mit++;
		if(mit != loaded.end())
			result.append(", ");
		mit--;
	}
	result.append("\n");

	lua_pushstring(L, result.c_str());
	return 1;
}



// Load a library into the process
//
// *****************************************************************
// all the following was broke. Idea was to load dependancies needed to
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
	int module_version = 0;
	
	if(loaded[name] != 0)//then already loaded
	{
		return 3;
	}
	string src_dir;
	
	
	_handle handle = 0;
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
		

		snprintf(buf, bufsize, "%s/%s.%s", mod_dirs[i].c_str(), name.c_str(), SO_EXT);
		
#ifndef WIN32
		dlerror();
		handle = dlopen(buf,  RTLD_NOW | RTLD_GLOBAL);
		// should this be reported? We may be able to deal with it. Need to think this through
// 		const char* dle = dlerror();
// 		if(dle)
// 		{
// 			fprintf(stderr, "dlsym error: `%s'\n", dle);
// 		}
#else
		WCHAR* str  = new WCHAR[bufsize];
		MultiByteToWideChar(0, 0, buf, bufsize-1, str, bufsize);
		handle = LoadLibrary(str);
		if(!handle)
		{
			fprintf(stderr, "Library failed to load\n");
		}
		delete [] str;
#endif

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

#ifndef WIN32
	lua_func lib_register = (lua_func) dlsym(handle, "lib_register");
	lua_func lib_version  = (lua_func) dlsym(handle, "lib_version");

	// don't know if we should report this, it may be resolved later
// 	const char* dle = dlerror();
// 	if(dle)
// 	{
// 		fprintf(stderr, "dlsym error: `%s'\n", dle);
// 	}
	if(!lib_register)
	{
		// we'll report this later 
		dlclose(handle);
		return 1;
    }
#else
	lua_func lib_register = (lua_func)GetProcAddress(handle, "lib_register");
	lua_func lib_version  = (lua_func)GetProcAddress(handle, "lib_version");
	
	if(!lib_register)
	{
		// we'll report this later 
		FreeLibrary(handle);
		return 1;
    }
#endif	
    
	lib_register(L); // call the register function from the library
	
	if(buf)
		delete [] buf;

// 	if(!suppress)
// 	{
// 		cout << "  Loading Module: " << name << endl;
		//cout << "    from " << src_dir << endl;
// 	}
	
	if(!lib_version)
	{
		printf("WARNING: Failed to load `lib_version' from `%s' setting version to -100\n", name.c_str());
		loaded[name] = -100;
		
	}
	else
	{
		loaded[name] = lib_version(L);
		if(loaded[name] == 0)
		{
			printf("WARNING: `lib_version' from `%s' returned 0. Changing version to -100\n", name.c_str());
			loaded[name] = -100;
		}
	}
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
	
#ifndef WIN32
	struct dirent *dp;
#else

	HANDLE hFind = INVALID_HANDLE_VALUE;
#endif

	// make a list of all the modules we want to load
	for(unsigned int d=0; d<mod_dirs.size(); d++)
	{
#ifndef WIN32
		DIR *dir = opendir(mod_dirs[d].c_str());
#else

		char nDir[MAX_PATH];
		_snprintf(nDir, MAX_PATH,  "%s\\*.%s", mod_dirs[d].c_str(), SO_EXT);
		
		std::string ns(nDir); //narrow string
		std::wstring ws(ns.length(), L' '); //wide string
		
		std::copy(ns.begin(), ns.end(), ws.begin());

		WIN32_FIND_DATA dir;

		hFind = FindFirstFile( ws.c_str(), &dir);
#endif

#ifndef WIN32
		if(dir)
		{
			while( (dp=readdir(dir)) != NULL)
#else
		if(hFind != INVALID_HANDLE_VALUE)
		{
			do
#endif
			{
#ifndef WIN32
				const char* filename = dp->d_name;
#else
				char filename[4096];
				wcstombs(filename, dir.cFileName, 4096);
#endif

				int len = strlen(filename);
				if(strncasecmp(filename+len-strlen(SO_EXT), SO_EXT, strlen(SO_EXT)) == 0) //then ends with so or dll 
				{
					const char ii = strlen(SO_EXT);
					char* n = new char[len+ii];
					strncpy(n, filename, len);
					n[len-ii-1] = 0;
					
					unloaded[n]++; //mark this module to be loaded
					
					delete [] n;
				}
			}
#ifdef WIN32
			while(FindNextFile(hFind, &dir) != 0);
			FindClose(hFind);
#else
			closedir(dir);
#endif
		}
		else
		{
#ifndef WIN32
			cerr << "Failed to read directory `" << mod_dirs[d] << "': " << strerror(errno) << endl;
#else
			cerr << "Failed to read directory `" << mod_dirs[d] << "': " << GetLastError() << endl;
#endif
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
	
	
	if(!suppress)
	{
		cout << "Modules: ";
		for(mit=loaded.begin(); mit!=loaded.end(); ++mit)
		{
			cout << (*mit).first;
			cout << "(r" << (*mit).second << ")";
			mit++;
			if( mit != loaded.end())
				cout << ", ";
			mit--;
		}
		cout << endl;
	}
	
#ifdef WIN32
#define dlerror() "dlerror"
#endif

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

const char* reference()
{
	return "Use the following reference when citing this code:\n" \
		   "*** reference incomplete ***\n" \
	       "\"MagLua, a Micromagnetics Programming Environment\". Mercer, Jason I. (2011). Journal. Vol, pages";
}

void print_help()
{
	cout << "MagLua r-" << __rev << " by Jason Mercer (c) 2011" << endl;
	cout << endl;
	cout << " MagLua is a micromagnetics programming environment built" << endl;
	cout << " on top of the Lua scripting language." << endl;
	cout << endl;
	cout << "Command Line Arguments:" << endl;
	cout << " -L mod_dir       Add module <mod_dir> to search path" << endl;
	cout << " -q               Run quietly, omit some startup messages" << endl;
	cout << " --module_path    Print primary module directory" << endl;
#ifdef WIN32
	cout << " --setup mod_dir  Setup startup files in $(APPDATA)\\maglua" << endl;
#else
	cout << " --setup mod_dir  Setup startup files in $(HOME)/.maglua.d" << endl;
#endif
	cout << "                   with <mod_dir> in the list of paths" << endl;
	cout << " -h, --help       Show this help" << endl;
	cout << endl;
	cout << reference() << endl;
	
	cout << endl;
}

