#include "libLuaSqlite.h"

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
	
	int rv = sqlite3_open(filename, &s);
	
	if(rv)
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
	
	lua_newtable(L);
	if(sqlite3_exec(sql, statement, exec_callback, &li, &errmsg))
		return luaL_error(L, errmsg);

	return 1;
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