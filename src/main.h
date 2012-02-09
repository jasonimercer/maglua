#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef MAGLUA_EXPORTS
  #define MAGLUA_API __declspec(dllexport)
 #else
  #define MAGLUA_API __declspec(dllimport)
 #endif
#else
 #define MAGLUA_API 
#endif
