#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)


 #ifdef LUAGRAPHICS_EXPORTS
  #define LUAGRAPHICS_API __declspec(dllexport)
 #else
  #define LUAGRAPHICS_API __declspec(dllimport)
 #endif
#else
 #define LUAGRAPHICS_API 
#endif
