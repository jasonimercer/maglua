#ifndef INLINE_IMPORTER
#define INLINE_IMPORTER

#ifdef WIN32
#include <string>
#include <windows.h>
#include <iostream>
#include <stdio.h>
#include <lmerr.h>


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
		int plen = path.length() + 10;
		char* pp = (char*)malloc(plen);
		sprintf_s(pp, plen, "%s", path.c_str());

		wchar_t* wpp = new wchar_t[plen];
		memset(wpp,0,plen);
		::MultiByteToWideChar(  CP_ACP, NULL,pp, -1, wpp,plen );

		
		int nlen = name.length() + 10;
		char* nn = (char*)malloc(nlen);
		sprintf_s(nn, plen, "%s", name.c_str());
		
		//fprintf(stderr, "%s   %s\n", pp, nn);

		//HINSTANCE handle = LoadLibraryA(path.c_str());
		//HINSTANCE handle = LoadLibraryA(pp);
		
		//SetErrorMode(SEM_NOGPFAULTERRORBOX);
		HINSTANCE handle = LoadLibraryEx( wpp, 0, 0);//LOAD_WITH_ALTERED_SEARCH_PATH);

		if(!handle) //try to fetch it as if it were already loaded (it may be)
		{
			set_lib_error(GetLastError(), error);
			//handle = GetModuleHandleA(path.c_str());
			handle = GetModuleHandleA(pp);
		}

		if(handle)
		{
			//func = (T) GetProcAddress(handle,name.c_str());
			func = (T) GetProcAddress(handle,nn);
		}

		free(pp);
		free(nn);
		delete [] wpp;
	}
	else
	{
		func = (T) GetProcAddress(GetModuleHandle(0), name.c_str()); //load from local space
	}

	if(func == 0)
	{
		std::string last_err = error;
		error = "Failed to load `";
		error.append(name);
		error.append("' from `");
		error.append(path);
		error.append("'");
		if(last_err.length())
		{
			error.append(": ");
			error.append(last_err);
		}
	}
	else
		error = "";
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
