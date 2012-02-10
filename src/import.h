#ifndef INLINE_IMPORTER
#define INLINE_IMPORTER

#ifdef WIN32
#include <string>
#include <windows.h>
#include <iostream>
#include <stdio.h>

template <class T>
inline T import_function_e(std::string path, std::string name, std::string& error)
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

	if(func == 0)
	{
		error = "Failed to load `";
		error.append(name);
		error.append("' from `");
		error.append(path);
		error.append("'");
	}
	return func;
}

template <class T>
inline T import_function(std::string path, std::string name)
{
	std::string e;
	return import_function_e<T>(path, name, e);
}
#else
#include <string>
#include <stdio.h>
#include <dlfcn.h>

template <class T>
inline T import_function_e(std::string path, std::string name, std::string& error)
{
	void* handle = dlopen(path.c_str(),  RTLD_NOW | RTLD_GLOBAL);
	T func = 0;
	if(handle)
	{
		func = (T) dlsym(handle, name.c_str());
		if(!func)
		{
			error = dlerror();
			dlclose(handle);
		}
	}
  	else
	{
		error = dlerror();
	}

	return func;
}

template <class T>
inline T import_function(std::string path, std::string name)
{
	std::string e;
	return import_function_e<T>(path, name, e);
}
#endif
#endif
