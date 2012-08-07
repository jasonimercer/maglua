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

#ifndef SQLITE3_CONN_H
#define SQLITE3_CONN_H

#include <sqlite3.h>
#include <string>
using namespace std;

#include "luabaseobject.h"

class SQLITE3_API sqlite3_conn : public LuaBaseObject
{
public:
	sqlite3_conn();
	virtual ~sqlite3_conn();
	
	LINEAGE1("SQLite3")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	virtual void push(lua_State* L);
	static int help(lua_State* L);

	void encode(buffer* b) {}
	int  decode(buffer* b) {return 0;}

	sqlite3* db;
	string filename;
};

#endif
