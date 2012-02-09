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

typedef int (*lua_func)(lua_State*);
typedef const char* (*c_lua_func)(lua_State*);

int lua_loadfile(lua_State* L)
{
	const string fullpath = lua_tostring(L, 1);
	int version = -100;
	string truename;
	
	string reg_error;
	
	  lua_func lib_register = import_function_e<  lua_func>(fullpath, "lib_register", reg_error);
	  lua_func lib_version  = import_function  <  lua_func>(fullpath, "lib_version");
  	c_lua_func lib_name     = import_function  <c_lua_func>(fullpath, "lib_name");
	  lua_func lib_main     = import_function  <  lua_func>(fullpath, "lib_main");
	  
	if(!lib_register)
	{
		lua_newtable(L);
		lua_pushstring(L, "error");
		lua_pushstring(L, reg_error.c_str());
		lua_settable(L, -3);
		// we'll report this later 
		return 1; //nil
    }

	if(lib_register(L)) // call the register function from the library
	{
		return luaL_error(L, "`lib_register' returned non-zero\n");
	}

	if(!lib_version)
	{
		version = -100;
	}
	else
	{
		version = lib_version(L);
		if(version == 0)
		{
			version = -100;
		}
	}
	
	if(!lib_name)
	{
		truename = "";
	}
	else
	{
		truename = lib_name(L);
	}
	
	if(lib_main)
	{
		if(lib_main(L))
			return luaL_error(L, "lib_main failed in `%s'\n", fullpath.c_str());
	}
	
	// ok, things are good?
	lua_newtable(L);
	lua_pushstring(L, "version");
	if(version == -100)
 		lua_pushinteger(L, 0);
	else
		lua_pushinteger(L, version);
	lua_settable(L, -3);
	
	lua_pushstring(L, "name");
	lua_pushstring(L, truename.c_str());
	lua_settable(L, -3);
	
	lua_pushstring(L, "fullpath");
	lua_pushstring(L, fullpath.c_str());
	lua_settable(L, -3);
	
	return 1;
}

