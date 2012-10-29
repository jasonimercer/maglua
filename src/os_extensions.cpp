#include "os_extensions.h"
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
 #define PATH_SEP "/"
#else
 #include <windows.h>
 #pragma warning(disable: 4251)
 #pragma warning(disable: 4996)
 #define snprintf _snprintf
 #define PATH_SEP "\\"
#endif


using namespace std;

static int l_os_pwd(lua_State* L)
{
#ifndef WIN32
	int s = 2048;
	int retry = 6;
	char* path = (char*)malloc(s);

	while(retry && !getcwd(path, s))
	{
		retry--;
		free(path);
		s *= 2;
		path = (char*)malloc(s);
	}

	if(!retry)
		return luaL_error(L, "Failed to get pwd");

	lua_pushstring(L, path);
	free(path);
#else

#endif
	return 1;
}

static int l_os_ls(lua_State* L)
{
	vector<string> files;
	string path; 
	if(!lua_isstring(L, 1))
	{
		path  = ".";
	}
	else
		path = lua_tostring(L, 1);

#ifndef WIN32
	struct dirent *dp;
#else
	HANDLE hFind = INVALID_HANDLE_VALUE;
#endif
	
#ifndef WIN32
	DIR *dir = opendir(path.c_str());
#else
	char nDir[MAX_PATH];
	_snprintf(nDir, MAX_PATH,  "%s\\*.*", path.c_str());
	
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
			string fullpath = path;
			fullpath.append(PATH_SEP);
			fullpath.append(filename);
			
			files.push_back(fullpath);
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
// / *
//   * on error we'll do nothing. An error means the directory doesn't exist, which is like it's empty
#ifndef WIN32
			return luaL_error(L, "Failed to read directory `%s':%s", path.c_str(), strerror(errno));
#else
			return luaL_error(L, "Failed to read directory `%s':%s", path.c_str(), GetLastError());
#endif
// */
	}
	
	lua_newtable(L);
	
	for(unsigned int i=0; i<files.size(); i++)
	{
		lua_pushinteger(L, i+1);
		lua_pushstring(L, files[i].c_str());
		lua_settable(L, -3);
	}
	return 1;
}



void register_os_extensions(lua_State* L)
{
	lua_getglobal(L, "os");

	lua_pushstring(L, "ls");
	lua_pushcfunction(L, l_os_ls);
	lua_settable(L, -3);

	lua_pushstring(L, "pwd");
	lua_pushcfunction(L, l_os_pwd);
	lua_settable(L, -3);

	lua_pop(L, 1);
}
