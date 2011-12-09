#ifndef INLINE_IMPORTER
#define INLINE_IMPORTER

#ifdef WIN32
#include <string>
#include <windows.h>
#include <iostream>
#include <stdio.h>

template <class T>
inline T import_function(std::string path, std::string name)
{
	T func = 0;
	if(path.length())
	{
		HINSTANCE handle = LoadLibraryA(path.c_str());

		if(!handle) //try to fetch it as if it were already loaded (it may be)
		{
			handle = GetModuleHandleA(path.c_str());
		}

		if(handle)
		{
			func = (T) GetProcAddress(handle,name.c_str());
		}
	}
	else
	{
		func = (T) GetProcAddress(GetModuleHandle(0), name.c_str()); //load from local space
	}
	return func;
}
#else
#include <string>
#include <stdio.h>
#include <dlfcn.h>
template <class T>
inline T import_function(std::string path, std::string name)
{
	void* handle = dlopen(path.c_str(),  RTLD_NOW | RTLD_GLOBAL);
	T func = 0;
	if(handle)
	{
		func = (T) dlsym(handle, name.c_str());
		if(!func)
		{
			printf("dlsym() %s\n", dlerror());
			dlclose(handle);
		}
	}
//  	else
//  		printf("dlsym() %s\n", dlerror());
	return func;
}

#endif
#endif
