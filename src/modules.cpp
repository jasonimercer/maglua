#include "modules.h"
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

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

int lua_getModulesInDirectory(lua_State* L)
{
	vector<string> module_paths; //this is where the filenames will go
	vector<string> module_dirs;
	if(!lua_isstring(L, 1))
	{
		return luaL_error(L, "need directory");
	}
	module_dirs.push_back(lua_tostring(L, 1)); // this single element vector is a hack to adapt old (working) code
	
	
#ifdef WIN32
	// need to locally add mod dirs to PATH so that it will 
	// automatically satisfy dll reqs. Hate Windows.
	// The (weak) assumption here is that if you're looking for 
	// modules in a directory then that directory should be in the path.
	// This depends on reasonable programming in bootstrap.lua, which 
	// in inaccessible to users. 
	{
		char* p = getenv("PATH");
		string sp = p;

		if( sp.find(module_dirs[0]) == string::npos ) //then PATH doesn't have path yet
		{
			sp.append(";");
			sp.append(module_dirs[0]);
			string patheq = "PATH=";
			patheq.append(sp);
			putenv(patheq.c_str());
		}
	}
#endif
	
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
/*
 * on error we'll do nothing. An error means the directory doesn't exist, which is like it's empty
#ifndef WIN32
			return luaL_error(L, "Failed to read directory `%s':%s", module_dirs[d].c_str(), strerror(errno));
#else
			return luaL_error(L, "Failed to read directory `%s':%s", module_dirs[d].c_str(), GetLastError());
#endif
*/
		}
	}
	
	lua_newtable(L);
	
	for(unsigned int i=0; i<module_paths.size(); i++)
	{
		lua_pushinteger(L, i+1);
		lua_pushstring(L, module_paths[i].c_str());
		lua_settable(L, -3);
	}
	return 1;
}

