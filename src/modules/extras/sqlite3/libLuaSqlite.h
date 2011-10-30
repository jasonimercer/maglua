extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef SQLITE3_EXPORTS
  #define SQLITE3_API __declspec(dllexport)
 #else
  #define SQLITE3_API __declspec(dllimport)
 #endif
#else
 #define SQLITE3_API 
#endif

