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
#ifdef WIN32
#ifdef ENCODE_EXPORTS
ENCODE_API Encodable* Factory_newItem(int id);
ENCODE_API void Factory_lua_pushItem(lua_State* L, Encodable* item, int id);
ENCODE_API int Factory_registerItem(int id, newFactoryFunction func, pushFunction Push, std::string name);
ENCODE_API void Factory_cleanup();
#else
#include <windows.h>
inline Encodable* Factory_newItem(int id)
{
	typedef Encodable* (*func)(int); 
	static func thefunc = 0;
	if(!thefunc)
	{
		thefunc = (func) GetProcAddress(GetModuleHandle(NULL), "Factory_newItem");
	}

	if(!thefunc)
	{
		printf("failed to load Factory_newItem\n");
		return NULL;
	}
	return thefunc(id);
}

inline void Factory_lua_pushItem(lua_State* L, Encodable* item, int id)
{
	typedef void (*func)(lua_State*, Encodable*, int); 
	static func thefunc = 0;
	if(!thefunc)
	{
		thefunc = (func) GetProcAddress(GetModuleHandle(NULL), "Factory_lua_pushItem");
	}

	if(!thefunc)
	{
		printf("failed to load Factory_lua_pushItem\n");
		return;
	}
	thefunc(L, item, id);
}

inline int Factory_registerItem(int id, newFactoryFunction f, pushFunction Push, std::string name)
{
	typedef int (*func)(int, newFactoryFunction, pushFunction, std::string); 
	static func thefunc = 0;
	if(!thefunc)
	{
		thefunc = (func) GetProcAddress(GetModuleHandle(NULL), "Factory_registerItem");
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

inline void Factory_cleanup()
{
	typedef void (*func)(); 
	static func thefunc = 0;
	if(!thefunc)
	{
		thefunc = (func) GetProcAddress(GetModuleHandle(NULL), "Factory_cleanup");
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
ENCODE_API Encodable* Factory_newItem(int id);
ENCODE_API void Factory_lua_pushItem(lua_State* L, Encodable* item, int id);
ENCODE_API int Factory_registerItem(int id, newFactoryFunction func, pushFunction Push, std::string name);
ENCODE_API void Factory_cleanup();
#endif


