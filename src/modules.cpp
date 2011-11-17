#include "modules.h"
#include <iostream>


#ifndef WIN32
 #include <stdlib.h>
 #include <string.h>
 #include <errno.h>
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


using namespace std;

void getModulePaths(lua_State* L, vector<string>& module_paths)
{
	module_paths.clear();

	vector<string> module_dirs;
	
	lua_getglobal(L, "module_path");
	lua_pushnil(L);
	while(lua_next(L, -2) != 0)
	{
		module_dirs.push_back(lua_tostring(L, -1));
		lua_pop(L, 1);
	}
	lua_pop(L, 1); //pop table
	
#ifndef WIN32
	struct dirent *dp;
#else
	HANDLE hFind = INVALID_HANDLE_VALUE;
#endif
	// make a list of all the modules we want to load
	for(unsigned int d=0; d<module_dirs.size(); d++)
	{
#ifndef WIN32
		DIR *dir = opendir(module_dirs[d].c_str());
#else
		char nDir[MAX_PATH];
		_snprintf(nDir, MAX_PATH,  "%s\\*.%s", module_dirs[d].c_str(), SO_EXT);
		
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
					n[len] = 0;

					string fullpath = module_dirs[d];
					
					fullpath.append(PATH_SEP);
					fullpath.append(n);
					
					module_paths.push_back(fullpath);
					
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
			cerr << "Failed to read directory `" << module_dirs[d] << "': " << strerror(errno) << endl;
#else
			cerr << "Failed to read directory `" << module_dirs[d] << "': " << GetLastError() << endl;
#endif
		}
		
	}
}



void getModuleDirectories(vector<string>& mds, vector<string>& initial_args)
{
	mds.clear();
	for(unsigned int i=0; i<initial_args.size(); i++)
	{
		if(	strcmp("-L", initial_args[i].c_str()) == 0 	&&
			i != initial_args.size()-2)
		{
			mds.push_back(initial_args[i+1]);
			i++;
		}
	}

	lua_State *L = lua_open();
	luaL_openlibs(L);
	
	lua_newtable(L);
	int j = 1;
	for(unsigned int i=1; i<initial_args.size(); i++) //skilling binary name
	{
		if(initial_args[i].size())
		{
			lua_pushinteger(L, j);
			lua_pushstring(L, initial_args[i].c_str());
			lua_settable(L, -3);
			j++;
		}
	}
	lua_setglobal(L, "arg");

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

	
	initial_args.clear();
	lua_getglobal(L, "arg");
	if(lua_istable(L, -1))
	{
		lua_pushnil(L);
		while(lua_next(L, -2))
		{
			if(lua_isstring(L, -1))
			{
				initial_args.push_back(lua_tostring(L, -1));
			}
			lua_pop(L, 1);
		}
	}
		
#ifdef WIN32
	free(home);
#endif
	lua_close(L);
}



