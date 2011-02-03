#include "libLuaSqlite.h"

#include <unistd.h>
#include <stdlib.h>

#define SLEEP_TIME (100000 + (rand() & 0x7FFFF))
#define RETRIES 100
sqlite3** checkSQLitep(lua_State* L, int idx)
{
	sqlite3** pp = (sqlite3**)luaL_checkudata(L, idx, "SQL");
    luaL_argcheck(L, pp != NULL, 1, "`SQL' expected");
	return pp;
}

sqlite3* checkSQLite(lua_State* L, int idx)
{
    return *checkSQLitep(L, idx);
}

void lua_pushSQLite(lua_State* L, sqlite3* sql)
{
	sqlite3** pp = (sqlite3**)lua_newuserdata(L, sizeof(sqlite3**));
	
	*pp = sql;
	luaL_getmetatable(L, "SQL");
	lua_setmetatable(L, -2);
}


int l_sql_new(lua_State* L)
{
	const char* filename = lua_tostring(L, 1);
	if(!filename)
		return luaL_error(L, "SQL.new requires a filename");
	
	char* errormsg = 0;
	sqlite3* s = 0;
	
	
	int retries = RETRIES;
	
	int rv = SQLITE_OK;
	for(int i=0; i<retries; i++)
	{
		rv = sqlite3_open(filename, &s);
		
		if(rv == SQLITE_BUSY)
		{
			int s = SLEEP_TIME;
// 			printf("%s is busy, sleeping for %i\n", filename, s);
			usleep(SLEEP_TIME); //sleep for a bit and then try again
		}
		else
		{
			break; //if rv != SQLITE_BUSY
		}
	}
	
	if(rv != SQLITE_OK)
	{
		if(s)
			return luaL_error(L, sqlite3_errmsg(s));
		else
			return luaL_error(L, "sqlite3 pointer is null");
	}	
	
	lua_pushSQLite(L, s);
	return 1;
}

int l_sql_tostring(lua_State* L)
{
	lua_pushstring(L, "SQL");
	return 1;	
}

typedef struct LI
{
	lua_State* L;
	int i;
} LI;

int l_sql_gc(lua_State* L)
{
	sqlite3* sql = checkSQLite(L, 1);
	if(!sql) return 0;

	sqlite3_close(sql);

	return 0;
}

int exec_callback(void* arg, int numcols, char** colvalues, char** colnames)
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

int l_sql_exec(lua_State* L)
{
	sqlite3* sql = checkSQLite(L, 1);
	if(!sql) return 0;

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
		int rv = sqlite3_exec(sql, statement, exec_callback, &li, &errmsg);
	
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
	
int l_sql_close(lua_State* L)
{
	sqlite3** sql = checkSQLitep(L, 1);
	if(!*sql) return 0;

	sqlite3_close(*sql);
	*sql = 0;
	return 0;
}

int l_sql_changes(lua_State* L)
{
	sqlite3** sql = checkSQLitep(L, 1);
	if(!*sql) return 0;
	
	lua_pushinteger(L, sqlite3_changes(*sql));
	return 1;
}

int registerSQLite(lua_State* L)
{
	static const struct luaL_reg methods [] = {
		{"__gc",         l_sql_gc},
		{"__tostring",   l_sql_tostring},
		{"exec",         l_sql_exec},
		{"exec",         l_sql_exec},
		{"close",        l_sql_close},
		{"changes",      l_sql_changes},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "SQL");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);
	lua_settable(L, -3);
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_sql_new},
		{"open",                l_sql_new},
		{NULL, NULL}
	};
		
	luaL_register(L, "SQL", functions);
	lua_pop(L,1);
}