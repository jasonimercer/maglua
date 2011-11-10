#include "import.h" //for import funcs
#include "loader.h"


#ifndef WIN32
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <dirent.h>
#define HOME "HOME"
#define SO_EXT "so"
#define MAGLUA_SETUP_PATH "/.maglua.d/module_path.lua"
#define PATH_SEP "/"
#else
 #include <windows.h>
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)
 #pragma warning(disable: 4996)
 #define snprintf _snprintf
 #define HOME "APPDATA"
 #define SO_EXT "dll"
 #define MAGLUA_SETUP_PATH "\\maglua\\module_path.lua"
 #define PATH_SEP "\\"
#endif


#include <iostream>
using namespace std;
	
loader_item::loader_item()
 : filename(""), fullpath(""), registered(false), truename(""), main_return(0), version(0)
{
}

loader_item::loader_item(const std::string _fullpath)
 : fullpath(_fullpath), registered(false), truename(""), main_return(0), version(0)
{
	// will try to derive filename from fullpath
	size_t start = -1;
	size_t last = -1;
	
	do
	{
		last = start;
		start = fullpath.find(PATH_SEP, last+1);
	}while(start != string::npos);
	
	filename = fullpath.substr(last+1);
	
	// now need to strip .SO_EXT
	string so_ext = SO_EXT;
	int tail = so_ext.size() + 1; //1 for dot
	
	filename.resize( filename.size() - tail);
}


loader_item::loader_item(const loader_item& ls)
{
	fullpath = ls.fullpath;
	filename = ls.filename;
	registered = ls.registered;
	truename = ls.truename;
	main_return = ls.main_return;
	version = ls.version;
}





typedef int (*lua_func)(lua_State*);
typedef const char* (*c_lua_func)(lua_State*);
typedef int (*lua_func_aa)(lua_State*, int, char**);

static int load_item(lua_State* L, loader_item& item, int argc, char** argv, bool& big_fail);
static int attempt_round(lua_State* L, vector<loader_item>& items, int argc, char** argv, bool& big_fail);




int load_items(lua_State* L, vector<loader_item>& items, int argc, char** argv, int quiet)
{
	bool big_fail = false;
	
	int total_loaded = 0;
	int num_this_round = 0;

	do
	{
		num_this_round = attempt_round(L, items, argc, argv, big_fail);
		
		if(big_fail)
			return -1;
		
		total_loaded += num_this_round;
	}while(num_this_round > 0);

	return total_loaded;
}

static int attempt_round(lua_State* L, vector<loader_item>& items, int argc, char** argv, bool& big_fail)
{
	int num_loaded = 0;
	vector<loader_item>::iterator it;
	for(it=items.begin(); it!=items.end(); ++it)
	{
		if(!(*it).registered )
		{
			if(!load_item(L, *it, argc, argv, big_fail))
			{
				num_loaded++;
			}
			if(big_fail)
				return 0;
		}
	}
	return num_loaded;
}


static int load_item(lua_State* L, loader_item& item, int argc, char** argv, bool& big_fail)
{
	const string& name = item.filename;
	
	  lua_func    lib_register = import_function<  lua_func   >(item.fullpath, "lib_register");
	  lua_func    lib_version  = import_function<  lua_func   >(item.fullpath, "lib_version");
  	c_lua_func    lib_name     = import_function<c_lua_func   >(item.fullpath, "lib_name");
	  lua_func_aa lib_main     = import_function<  lua_func_aa>(item.fullpath, "lib_main");
	  
	if(!lib_register)
	{
		// we'll report this later 
#ifdef LOAD_LIB_DEBUG
		printf("(%s:%i) !lib_register\n", __FILE__, __LINE__);
#endif
		return 1;
    }
#ifdef LOAD_LIB_DEBUG
    else
		printf("(%s:%i) lib_register FOUND\n", __FILE__, __LINE__);
#endif

	if(lib_register(L)) // call the register function from the library
	{
		item.error = "`lib_register' returned non-zero";
		// the windows version of the code has some hackish just-in-time dynamic linking
		// when it fails (ie. Encode hasn't loaded yet), lib_register will return non-zero
		printf("lib_register returned non-zero (%s:%i)\n", __FILE__, __LINE__);
		return 2;
	}
	item.registered = true;

	if(!lib_version)
	{
		item.error = "Failed to load `lib_version'";
		printf("WARNING: Failed to load `lib_version' from `%s' setting version to -100\n", name.c_str());
		item.version = -100;
	}
	else
	{
		item.version = lib_version(L);
		if(item.version == 0)
		{
			printf("WARNING: `lib_version' from `%s' returned 0. Changing version to -100\n", name.c_str());
			item.version = -100;
		}
	}
	
	if(!lib_name)
	{
		printf("WARNING: Failed to load `lib_name' from `%s' setting truename to path\n", name.c_str());
		item.truename = name;
	}
	else
	{
		item.truename = lib_name(L);
	}
	
	big_fail = false;
	if(lib_main)
	{
		if(lib_main(L, argc, argv))
			big_fail = true;
	}

	return 0;
}
