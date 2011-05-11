#include <sqlite3.h>
#include "libLuaSqlite.h"

#include <unistd.h>
#include <stdlib.h>

#include <string>
#include <map>
#include <iostream>
#include <semaphore.h>

using namespace std;

static map<string,sem_t> semLookup;

static void waitDB(string name)
{
	if(semLookup.find(name) == semLookup.end()) //doesn't exist
	{
		sem_t& sem = semLookup[name];
		sem_init(&sem, 0, 1);
		
		cout << "New semaphore made for `" << name << "'" << endl;
	}
	
	sem_wait(&semLookup[name]);
}


static void postDB(string name)
{
	if(semLookup.find(name) == semLookup.end()) //doesn't exist
	{
		return;
	}
	sem_post(&semLookup[name]);
}


typedef struct sqlite3_conn
{
	sqlite3* db;
	string filename;
	int refcount;
} sqlite3_conn;

typedef struct LI
{
	lua_State* L;
	int i;
} LI;

#define SLEEP_TIME (100000 + (rand() & 0x7FFFF))
#define RETRIES 100

int lua_isSQLite(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "SQL");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

sqlite3_conn* lua_toSQLite(lua_State* L, int idx)
{
	sqlite3_conn** pp = (sqlite3_conn**)luaL_checkudata(L, idx, "SQL");
	luaL_argcheck(L, pp != NULL, idx, "`SQL' expected");
	return *pp;
}

void lua_pushSQLite(lua_State* L, sqlite3_conn* s)
{
	sqlite3_conn** pp = (sqlite3_conn**)lua_newuserdata(L, sizeof(sqlite3_conn**));
	
	*pp = s;
	luaL_getmetatable(L, "SQL");
	lua_setmetatable(L, -2);
	s->refcount++;
}


int l_sql_new(lua_State* L)
{
	const char* filename = lua_tostring(L, 1);
	if(!filename)
		return luaL_error(L, "SQL.new requires a filename");
	
	char* errormsg = 0;
// 	sqlite3* s = 0;
	
	sqlite3_conn* sql = new sqlite3_conn();
	sql->refcount = 0;
	sql->filename = filename;
	
	int retries = RETRIES;
	
	int rv = SQLITE_OK;
	for(int i=0; i<retries; i++)
	{
		waitDB(filename);
		rv = sqlite3_open(filename, &(sql->db));
		postDB(filename);
		
		if(rv == SQLITE_BUSY)
		{
			int s = SLEEP_TIME;
			printf("%s is busy, sleeping for %i\n", filename, s);
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
		if(sql->db)
		{
			int r = luaL_error(L, sqlite3_errmsg(sql->db));
			delete sql;
			return r;
		}
		else
		{
			delete sql;
			return luaL_error(L, "sqlite3 pointer is null");
		}
	}
	
	
	lua_pushSQLite(L, sql);
	return 1;
}

static int l_sql_tostring(lua_State* L)
{
	lua_pushstring(L, "SQL");
	return 1;	
}



static int l_sql_gc(lua_State* L)
{
	sqlite3_conn* sql = lua_toSQLite(L, 1);
	if(!sql) return 0;
	if(!sql->db) return 0;
	
	sqlite3_close(sql->db);

	return 0;
}

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
	sqlite3_conn* sql = lua_toSQLite(L, 1);
	if(!sql) return 0;
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
	sqlite3_conn* sql = lua_toSQLite(L, 1);
	if(!sql) return 0;
	if(!sql->db) return luaL_error(L, "Database is not opened");
	
	sqlite3_close(sql->db);
	sql->db = 0;
	return 0;
}

static int l_sql_changes(lua_State* L)
{
	sqlite3_conn* sql = lua_toSQLite(L, 1);
	if(!sql) return 0;
	if(!sql->db) return luaL_error(L, "Database is not opened");
	
	lua_pushinteger(L, sqlite3_changes(sql->db));
	return 1;
}

int registerSQLite(lua_State* L)
{
	static const struct luaL_reg methods [] = {
		{"__gc",         l_sql_gc},
		{"__tostring",   l_sql_tostring},
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