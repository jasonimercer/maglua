#include "libLuaMigrate.h"
#include <string.h>
#include <stdlib.h>

void addToBuff(char** buf, int* bsize, int* blen, const char* data, int dlen)
{
	if(*blen + dlen > *bsize) //need to expand buffer
	{
		*bsize *= 2;
		*bsize += dlen;
		*buf = (char*)realloc(*buf, *bsize);
	}

	memcpy(*buf+*blen, data, dlen);
	*blen += dlen;
}

static int lexportwriter(lua_State *L, const void* b, size_t size, void* d) 
{
	(void)L;
	lua_Variable* v = (lua_Variable*)d;

	addToBuff(&v->chunk, &v->chunksize, &v->chunklength, (const char*)b, size);

	return 0;
}

const char* limportreader(lua_State* L, void* data, size_t* size)
{
	(void)L;
	lua_Variable* v = (lua_Variable*)data;

	*size = v->chunklength;

	v->chunklength = 0;

	return v->chunk;
}

void initLuaVariable(lua_Variable* v)
{
	v->type = LUA_TNIL;

	v->chunk  = 0;
	v->chunksize  = 0;
	v->chunklength= 0;

	v->listKey = 0;
	v->listVal = 0;
	v->listlength = 0;
}

void freeLuaVariable(lua_Variable* v)
{
	if(v->chunk)
		free(v->chunk);
	v->chunk = 0;
	
	for(int i=0; i<v->listlength; i++)
	{
		freeLuaVariable(&v->listKey[i]);
		freeLuaVariable(&v->listVal[i]);
	}

	if(v->listlength)
	{
		free(v->listKey);
		free(v->listVal);
	}
}

void importLuaVariable(lua_State* L, lua_Variable* v)
{
	int i, j;
	switch(v->type)
	{
		case LUA_TNIL:
			lua_pushnil(L);
		break;
		case LUA_TBOOLEAN:
			lua_pushboolean(L, *((double*)v->chunk)>0.5);
		break;
		case LUA_TLIGHTUSERDATA:
			luaL_error(L, "Cannot import LIGHTUSERDATA");
		break;
		case LUA_TNUMBER:
			lua_pushnumber(L, *((double*)v->chunk));
		break;
		case LUA_TSTRING:
			lua_pushstring(L, v->chunk);
		break;
		case LUA_TTABLE:
			lua_newtable(L);
			j = lua_gettop(L);
			for(i=0; i<v->listlength; i++)
			{
				importLuaVariable(L, &v->listKey[i]);
				importLuaVariable(L, &v->listVal[i]);
				lua_settable(L, j);
			}
		break;
		case LUA_TFUNCTION:
			i = lua_load(L, limportreader, v, "import_function");
			if(i)
				printf("%s\n", lua_tostring(L, -1));
		break;
		case LUA_TUSERDATA:
			luaL_error(L, "Cannot import USERDATA");
		break;
		case LUA_TTHREAD:
			luaL_error(L, "Cannot import THREAD");
		break;

		default:
			luaL_error(L, "Unknown lua type: %i", v->type);
			fprintf(stderr, "Unknown lua type: %i", v->type);
	}
}

void exportLuaVariable(lua_State* L, int index, lua_Variable* v)
{
	v->type = lua_type(L, index);
	int tablesize;
	const char* c;
	
	if(index < 0)
		index = lua_gettop(L) + index + 1;

#if 0
	printf("\nExport:\n");
 	for(int i=lua_gettop(L); i>0; i--)
	{
		if(i == index)
			printf("**");
		else
			printf("  ");
 		printf(" %2i) %s  ", i, lua_typename(L, lua_type(L, i)));
		lua_getglobal(L, "tostring");
		lua_pushvalue(L, i);
		lua_call(L, 1, 1);
		printf(" '%s'\n", lua_tostring(L, -1));
		lua_pop(L, 1);
	}

 	//printf("export (%i): %s\n", index, lua_typename(L, v->type));
#endif

	switch(v->type)
	{
		case LUA_TNIL:
		break;
		case LUA_TBOOLEAN:
		{
			double d = lua_toboolean(L, index);
			addToBuff(&(v->chunk), &(v->chunksize), &(v->chunklength), (const char*)&d, sizeof(double));
		}
		break;
		case LUA_TLIGHTUSERDATA:
			luaL_error(L, "Cannot export LIGHTUSERDATA");
		break;
		case LUA_TNUMBER:
		{
			double d = lua_tonumber(L, index);
			addToBuff(&(v->chunk), &(v->chunksize), &(v->chunklength), (const char*)&d, sizeof(double));
		}
		break;
		case LUA_TSTRING:
			c = lua_tostring(L, index);
			addToBuff(&(v->chunk), &(v->chunksize), &(v->chunklength), c, strlen(c)+1);
		break;
		case LUA_TTABLE:
			tablesize = 0;
			lua_pushnil(L);
			while(lua_next(L, index) != 0)
			{
				tablesize++;
				lua_pop(L, 1);
			}

			v->listlength = tablesize;
			v->listKey = (lua_Variable*)malloc(sizeof(lua_Variable) * tablesize);
			v->listVal = (lua_Variable*)malloc(sizeof(lua_Variable) * tablesize);
			tablesize = 0;

			lua_pushnil(L);
			while(lua_next(L, index) != 0)
			{
				initLuaVariable(&v->listKey[tablesize]);
				initLuaVariable(&v->listVal[tablesize]);
				exportLuaVariable(L, lua_gettop(L)-1, & v->listKey[tablesize]);
				exportLuaVariable(L, lua_gettop(L),   & v->listVal[tablesize]);
				tablesize++;
				lua_pop(L, 1);
			}
		break;
		case LUA_TFUNCTION:
			lua_pushvalue(L, index); //copy func to top
			if(lua_dump(L, lexportwriter, v) != 0)
				luaL_error(L, "Unable to upload function");
			lua_pop(L, 1);
		break;
		case LUA_TUSERDATA:
			luaL_error(L, "Cannot export USERDATA");
		break;
		case LUA_TTHREAD:
			luaL_error(L, "Cannot export THREAD");
		break;
	}
}
