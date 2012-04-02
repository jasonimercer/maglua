extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

class LuaBaseObject;

typedef LuaBaseObject*(*newFactoryFunction)();
typedef void (*pushFunction)(lua_State*, LuaBaseObject*);


#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef LUABASEOBJECT_EXPORTS
  #define LUABASEOBJECT_API __declspec(dllexport)
 #else
  #define LUABASEOBJECT_API __declspec(dllimport)
 #endif
#else
 #define LUABASEOBJECT_API 
#endif

extern "C"
{
LUABASEOBJECT_API LuaBaseObject* Factory_newItem(int id);
LUABASEOBJECT_API void Factory_lua_pushItem(lua_State* L, LuaBaseObject* item, int id);
LUABASEOBJECT_API int Factory_registerItem(int id, newFactoryFunction func, pushFunction Push, const char* name);
LUABASEOBJECT_API void Factory_cleanup();
LUABASEOBJECT_API int hash32(const char* string);
}

