#ifndef INLINE_IMPORTER
#define INLINE_IMPORTER

#ifdef WIN32
#include <string>
#include <windows.h>
#include <iostream>
#include <stdio.h>
#include <lmerr.h>
using namespace std;

inline void set_lib_error(DWORD dwLastError, std::string& error)
{
    LPSTR MessageBuffer;
    DWORD dwBufferLength;

    DWORD dwFormatFlags = FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_IGNORE_INSERTS |
        FORMAT_MESSAGE_FROM_SYSTEM ;
    if(dwBufferLength = FormatMessageA(
        dwFormatFlags,
        NULL,
        dwLastError,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR) &MessageBuffer,
        0,
        NULL
        ))
    {
		error = MessageBuffer;
        LocalFree(MessageBuffer);
    }
}

template <class T>
inline T import_function_e(std::string path, std::string name, std::string& error)
{
	T func = 0;
	if(path.length())
	{
		//cout << path << endl;
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
		std::string foo;
		set_lib_error(GetLastError(), foo);

		error = "Failed to load `";
		error.append(name);
		error.append("' from `");
		error.append(path);
		error.append("'");
		
		if(foo.length())
		{
			error.append(": ");
			error.append(foo);
		}
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
