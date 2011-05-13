#include "libLuaMysql.h"

#include <unistd.h>
#include <stdlib.h>

#include <iostream>
#include <vector>
#include <string>

using namespace std;

typedef struct mysql_conn
{
	MYSQL db;
	int refcount;
	int num_rows;
} mysql_conn;


int lua_isMySQL(lua_State* L, int idx)
{
	lua_getmetatable(L, idx);
	luaL_getmetatable(L, "MySQL");
	int eq = lua_equal(L, -2, -1);
	lua_pop(L, 2);
	return eq;
}

mysql_conn* lua_toMySQL(lua_State* L, int idx)
{
	mysql_conn** pp = (mysql_conn**)luaL_checkudata(L, idx, "MySQL");
	luaL_argcheck(L, pp != NULL, idx, "`MySQL' expected");
	return *pp;
}

void lua_pushMySQL(lua_State* L, mysql_conn* s)
{
	mysql_conn** pp = (mysql_conn**)lua_newuserdata(L, sizeof(mysql_conn**));
	
	*pp = s;
	luaL_getmetatable(L, "MySQL");
	lua_setmetatable(L, -2);
	s->refcount++;
}


static int l_new(lua_State* L)
{
	if(lua_gettop(L) < 4)
		return luaL_error(L, "MySQL.new requires a host, user, passwd and dbname");
	
	mysql_conn* sql = new mysql_conn();

	sql->refcount = 0;
	sql->num_rows = 0;
	
	
	if(!mysql_init(&sql->db))
	{
		delete sql;
		return luaL_error(L, "Failed to initialize MySQL");
	}
	
	const char* host = lua_tostring(L, 1);
	const char* user = lua_tostring(L, 2);
	const char* passwd = lua_tostring(L, 3);
	const char* dbname = lua_tostring(L, 4);
	
	if(!mysql_real_connect(&(sql->db), host, user, passwd, dbname, 0, 0, CLIENT_MULTI_STATEMENTS))
	{
		return luaL_error(L, "connect error: %s",  mysql_error(&(sql->db)));
	}
	
	lua_pushMySQL(L, sql);
	return 1;
}

static int l_tostring(lua_State* L)
{
	lua_pushstring(L, "MySQL");
	return 1;	
}



static int l_gc(lua_State* L)
{
	mysql_conn* sql = lua_toMySQL(L, 1);
	if(!sql) return 0;
	
	mysql_close(&(sql->db));
	delete sql;
	return 0;
}

static int l_exec(lua_State* L)
{
	mysql_conn* sql = lua_toMySQL(L, 1);
	if(!sql) return 0;
	MYSQL& mysql = sql->db;
	
	const char* statement = lua_tostring(L, 2);

	if(!statement)
		return luaL_error(L, "statement is null");

	if(mysql_query(&mysql, statement))
	{
		return luaL_error(L, "%s",  mysql_error(&mysql));
	}
	
	sql->num_rows = mysql_affected_rows(&mysql);
	
	int num_ret = 0;
	
	do
	{
		MYSQL_RES *result = mysql_store_result(&mysql);
		if (result)
		{
			unsigned int num_fields;
			unsigned int num_rows;
			unsigned int i;
			unsigned int rr = 1;

			MYSQL_FIELD *fields;
			MYSQL_ROW row;
			
			num_fields = mysql_num_fields(result);
			num_rows   = mysql_num_rows(result);
			fields     = mysql_fetch_fields(result);

			if(num_fields > 0)
			{
				lua_newtable(L);
				num_ret++;
			}
			
			while((row = mysql_fetch_row(result)))
			{
				lua_pushinteger(L, rr); rr++;
				
// 				unsigned long *lengths;
// 				lengths = mysql_fetch_lengths(result);
				
				lua_newtable(L);
				for(i = 0; i < num_fields; i++)
				{
					lua_pushstring(L, fields[i].name);
					lua_pushfstring(L, "%s", row[i] ? row[i] : "NULL");
					lua_settable(L, -3);
				}
				lua_settable(L, -3);
			}

			mysql_free_result(result);
		}
	}while(!mysql_next_result(&mysql));
    
	return num_ret;
}
	
// static int l_close(lua_State* L)
// {
// 	mysql_conn* sql = lua_toMySQL(L, 1);
// 	if(!sql) return 0;
// 	if(!sql->db) return luaL_error(L, "Database is not opened");
// 	
// 	sqlite3_close(sql->db);
// 	sql->db = 0;
// 	return 0;
// }

static int l_changes(lua_State* L)
{
	mysql_conn* sql = lua_toMySQL(L, 1);
	if(!sql) return 0;
	
	lua_pushinteger(L, sql->num_rows);
	return 1;
}

int registerMySQL(lua_State* L)
{
	static const struct luaL_reg methods [] = {
		{"__gc",         l_gc},
		{"__tostring",   l_tostring},
		{"exec",         l_exec},
// 		{"close",        l_close},
		{"changes",      l_changes},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MySQL");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);
	lua_settable(L, -3);
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_new},
		{"open",                l_new},
		{NULL, NULL}
	};
		
	
	luaL_register(L, "MySQL", functions);
	lua_pop(L,1);
}
