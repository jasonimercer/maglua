extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef ENCODE_EXPORTS
  #define ENCODE_API __declspec(dllexport)
 #else
  #define ENCODE_API __declspec(dllimport)
 #endif
#else
 #define ENCODE_API 
#endif

class Encodable;

typedef Encodable*(*newFactoryFunction)();
typedef void (*pushFunction)(lua_State*, Encodable*);


#include <string>
#include "main.h" //to get dynamic load  function
#ifdef WIN32
#ifdef ENCODE_EXPORTS
extern "C"
{
ENCODE_API Encodable* Factory_newItem(int id);
ENCODE_API void Factory_lua_pushItem(lua_State* L, Encodable* item, int id);
ENCODE_API int Factory_registerItem(int id, newFactoryFunction func, pushFunction Push, const char* name);
ENCODE_API void Factory_cleanup();
}
#else
#include <windows.h>
#include <iostream>
#include <string>
#include <stdio.h>

static Encodable* Factory_newItem(int id)
{
	typedef Encodable* (*func)(int); 
	static func thefunc = 0;

	if(!thefunc)
	{
		// first need to get function to get lib paths
		typedef const char* (*sfuncs) (const char*);
		sfuncs getPath = import_function<sfuncs>("", "get_libpath");

		if(!getPath)
		{
			fprintf(stderr, "Failed to load `get_libpath'\n");
			return 0;
		}

		const char* encPath = getPath("encode");

		if(!encPath)
			return 0;

		thefunc = import_function<func>(encPath, "Factory_newItem");
	}

	if(!thefunc)
	{
		printf("failed to load Factory_newItem\n");
		return NULL;
	}
	return thefunc(id);
}

static void Factory_lua_pushItem(lua_State* L, Encodable* item, int id)
{
	typedef void (*func)(lua_State*, Encodable*, int); 
	static func thefunc = 0;

	if(!thefunc)
	{
		// first need to get function to get lib paths
		typedef const char* (*sfuncs) (const char*);
		sfuncs getPath = import_function<sfuncs>("", "get_libpath");

		if(!getPath)
		{
			fprintf(stderr, "Failed to load `get_libpath'\n");
			return;
		}

		const char* encPath = getPath("encode");

		if(!encPath)
			return;

		thefunc = import_function<func>(encPath, "Factory_newItem");
	}

	if(!thefunc)
	{
		printf("failed to load Factory_lua_pushItem\n");
		return;
	}
	thefunc(L, item, id);
}

static int Factory_registerItem(int id, newFactoryFunction f, pushFunction Push, const char* name)
{
	typedef int (*func)(int, newFactoryFunction, pushFunction, const char*); 
	static func thefunc = 0;

	if(!thefunc)
	{
		// first need to get function to get lib paths
		typedef const char* (*sfuncs) (const char*);
		sfuncs getPath = import_function<sfuncs>("", "get_libpath");

		if(!getPath)
		{
			fprintf(stderr, "Failed to load `get_libpath'\n");
			return -1;
		}

		const char* encPath = getPath("encode");

		if(!encPath)
			return -1;

		thefunc = import_function<func>(encPath, "Factory_registerItem");
	}

	if(!thefunc)
	{
		// this may fail when encode.dll hasn't been loaded yet. 
		// so we'll return non-zero to try to load the calling resource
		// later
		//printf("failed to load Factory_registerItem\n");
		return -1;
	}

	return thefunc(id, f, Push, name);
}

static void Factory_cleanup()
{
	typedef void (*func)(); 
	static func thefunc = 0;
	if(!thefunc)
	{
		// first need to get function to get lib paths
		typedef const char* (*sfuncs) (const char*);
		sfuncs getPath = import_function<sfuncs>("", "get_libpath");

		if(!getPath)
		{
			fprintf(stderr, "Failed to load `get_libpath'\n");
			return;
		}

		const char* encPath = getPath("encode");

		if(!encPath)
			return;

		thefunc = import_function<func>(encPath, "Factory_newItem");
	}

	if(!thefunc)
	{
		printf("failed to load Factory_cleanup\n");
		return;
	}
	thefunc();
}
#endif
#else
extern "C"
{
ENCODE_API Encodable* Factory_newItem(int id);
ENCODE_API void Factory_lua_pushItem(lua_State* L, Encodable* item, int id);
ENCODE_API int Factory_registerItem(int id, newFactoryFunction func, pushFunction Push, const char* name);
ENCODE_API void Factory_cleanup();
}
#endif


