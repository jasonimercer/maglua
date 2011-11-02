extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

class Encodable;

typedef Encodable*(*newFactoryFunction)();
typedef void (*pushFunction)(lua_State*, Encodable*);


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


extern "C"
{
ENCODE_API Encodable* Factory_newItem(int id);
ENCODE_API void Factory_lua_pushItem(lua_State* L, Encodable* item, int id);
ENCODE_API int Factory_registerItem(int id, newFactoryFunction func, pushFunction Push, const char* name);
ENCODE_API void Factory_cleanup();
}
