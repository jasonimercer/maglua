#include "libLuaSqlite.h"

#include <unistd.h>
#include <stdlib.h>

#include <string>
#include <map>
#include <iostream>
#include <semaphore.h>

#define SLEEP_TIME (100000 + (rand() & 0x7FFFF))
#define RETRIES 100

using namespace std;

static map<string,sem_t> semLookup;


static void waitDB(const string& name)
{
	if(semLookup.find(name) == semLookup.end()) //doesn't exist
	{
		sem_t& sem = semLookup[name];
		sem_init(&sem, 0, 1);
		
		//cout << "New semaphore made for `" << name << "'" << endl;
	}
	
	sem_wait(&semLookup[name]);
}


static void postDB(const string& name)
{
	if(semLookup.find(name) == semLookup.end()) //doesn't exist
	{
		return;
	}
	sem_post(&semLookup[name]);
}



sqlite3_conn::sqlite3_conn()
	: LuaBaseObject(hash32(lineage(0)))
{
	db = 0;
	filename = "";
}

sqlite3_conn::~sqlite3_conn()
{
	if(db)
		sqlite3_close(db);
	db = 0;
}

int sqlite3_conn::luaInit(lua_State* L)
{
	const char* cfilename = lua_tostring(L, 1);
	if(!cfilename)
		return luaL_error(L, "SQLite3.new requires a filename");
	
	char* errormsg = 0;
	
	filename = cfilename;
	
	int retries = RETRIES;
	
	int rv = SQLITE_OK;
	for(int i=0; i<retries; i++)
	{
		waitDB(filename);
		rv = sqlite3_open(cfilename, &db);
		postDB(filename);
		
		if(rv == SQLITE_BUSY)
		{
			int s = SLEEP_TIME;
			printf("%s is busy, sleeping for %i\n", cfilename, s);
			usleep(s); //sleep for a bit and then try again
		}
		else
		{
			break; //if rv != SQLITE_BUSY
		}
	}
	
	//probably a leak here
	if(rv != SQLITE_OK)
	{
		if(db)
		{
			return luaL_error(L, sqlite3_errmsg(db));
		}
		else
		{
			return luaL_error(L, "sqlite3 pointer is null");
		}
	}
	
	return LuaBaseObject::luaInit(L);
}


typedef struct LI
{
	lua_State* L;
	int i;
} LI;



static int exec_callback(void* arg, int numcols, char** colvalues, char** colnames)
{
	LI* li = (LI*)arg;
	lua_State* L = li->L;
		
	lua_pushinteger(L, li->i);
	lua_newtable(L);
	
	for(int i=0; i<numcols; i++)
	{
		lua_pushstring(L, colnames[i]);
		lua_pushstring(L, colvalues[i]);
		lua_settable(L, -3);
	}
	
	lua_settable(L, -3);
	li->i++;
	return 0;
}

static int l_sql_exec(lua_State* L)
{
	LUA_PREAMBLE(sqlite3_conn, sql, 1);	

	if(!sql->db) return luaL_error(L, "Database is not opened");

	const char* statement = lua_tostring(L, 2);
	if(!statement)
		return 0;
	char* errmsg = 0;

	LI li;
	li.L = L;
	li.i = 1;
	
	int retries = RETRIES;
	
	lua_newtable(L);
	
	for(int i=0; i<retries; i++)
	{
		errmsg = 0;
		waitDB(sql->filename);
		int rv = sqlite3_exec(sql->db, statement, exec_callback, &li, &errmsg);
		postDB(sql->filename);
		
		if(rv == SQLITE_BUSY)
		{
			int s = SLEEP_TIME;
// 			printf("db is busy, sleeping for %i (%i)\n", s, i);
			usleep(s); //sleep for a bit and then try again
		}
		else
		{
			if(rv)
				return luaL_error(L, errmsg);
			
			if(rv == SQLITE_OK)
				return 1;
		}
	}
	
	if(errmsg)
		return luaL_error(L, errmsg);
	return luaL_error(L, "Failed to execute statement");
}
	
static int l_sql_close(lua_State* L)
{
	LUA_PREAMBLE(sqlite3_conn, sql, 1);	

	if(!sql->db) return 0;//luaL_error(L, "Database is not opened");
	
	sqlite3_close(sql->db);
	sql->db = 0;
	return 0;
}

static int l_sql_changes(lua_State* L)
{
	LUA_PREAMBLE(sqlite3_conn, sql, 1);	

	if(!sql->db) //return luaL_error(L, "Database is not opened");
		lua_pushinteger(L, 0);
	else
		lua_pushinteger(L, sqlite3_changes(sql->db));
	return 1;
}


static int l_escapestring(lua_State* L)
{
	LUA_PREAMBLE(sqlite3_conn, sql, 1);	
	
	const char* s = lua_tostring(L, 2);

	char* es = sqlite3_mprintf("%q",s);
	
	lua_pushstring(L, es);
	
	sqlite3_free(es);
	
	return 1;
}


int sqlite3_conn::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "SQLite3 database interaction object");
		lua_pushstring(L, "1 String: Filename of the database to open or create."); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	if(!lua_isfunction(L, 1))
	{
		return luaL_error(L, "help expect zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if(func == l_sql_exec)
	{
		lua_pushstring(L, "Execute a properly formatted SQLite3 instruction");
		lua_pushstring(L, "1 String: SQL instruction");
		lua_pushstring(L, "1 Table: Result of the operation.");
		return 3;
	}

	if(func == l_escapestring)
	{
		lua_pushstring(L, "Escape quotes in a string so that it may be used in an SQL statement");
		lua_pushstring(L, "1 String: Unescaped string.");
		lua_pushstring(L, "1 String: Escaped string.");
		return 3;
	}
	
	if(func == l_sql_close)
	{
		lua_pushstring(L, "Close the opened SQLite3 database");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_sql_changes)
	{
		lua_pushstring(L, "Determine how many records were changed by the most recent SQL operation");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Number of records changed.");
		return 3;
	}
		
	return LuaBaseObject::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* sqlite3_conn::luaMethods()
{
	if(m[127].name)
		return m;

	static const luaL_Reg _m[] =
	{
		{"exec",         l_sql_exec},
		{"close",        l_sql_close},
		{"changes",      l_sql_changes},
		{"escapeString", l_escapestring},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}

#include "info.h"
extern "C"
{
SQLITE3_API int lib_register(lua_State* L);
SQLITE3_API int lib_version(lua_State* L);
SQLITE3_API const char* lib_name(lua_State* L);
SQLITE3_API int lib_main(lua_State* L);
}



static int l_getmetatable(lua_State* L)
{
    if(!lua_isstring(L, 1))
        return luaL_error(L, "First argument must be a metatable name");
    luaL_getmetatable(L, lua_tostring(L, 1));
    return 1;
}

#include "sqlite_luafuncs.h"
SQLITE3_API int lib_register(lua_State* L)
{
	luaT_register<sqlite3_conn>(L);

    lua_pushcfunction(L, l_getmetatable);
    lua_setglobal(L, "maglua_getmetatable");
    if(luaL_dostring(L, __sqlite_luafuncs()))
    {
        fprintf(stderr, "%s\n", lua_tostring(L, -1));
        return luaL_error(L, lua_tostring(L, -1));
    }

    lua_pushnil(L);
    lua_setglobal(L, "maglua_getmetatable");

	return 0;
}

SQLITE3_API int lib_version(lua_State* L)
{
	return __revi;
}

SQLITE3_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "SQLite3";
#else
	return "SQLite3-Debug";
#endif
}

SQLITE3_API int lib_main(lua_State* L)
{
	return 0;
}
