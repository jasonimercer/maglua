#include "luamigrate.h"
#include <string.h>
#include <stdlib.h>

typedef struct buffer
{
	char* buf;
	int pos;
	int size;
}buffer;

void ensureSize(int add, buffer* b)
{
	if(b->pos + add >= b->size)
	{
		if(b->size)
		{
			b->size *= 2;
			if(b->pos + add >= b->size)
				b->size += add;
		}
		else
			b->size += add;
		b->buf = (char*)realloc(b->buf, b->size);
	}
}

void encodeBuffer(const void* s, int len, buffer* b)
{
	ensureSize(len, b);
	memcpy(b->buf + b->pos, s, len);
	b->pos += len;
}

void encodeDouble(const double d, buffer* b)
{
	encodeBuffer(&d, sizeof(d), b);
}

void encodeInteger(const int i, buffer* b)
{
	encodeBuffer(&i, sizeof(i), b);
}

int decodeInteger(const char* buf, int* pos)
{
	int i;
	memcpy(&i, buf+*pos, sizeof(int));
	(*pos) += sizeof(int);
	return i;
}
double decodeDouble(const char* buf, int* pos)
{
	double d;
	memcpy(&d, buf+*pos, sizeof(double));
	(*pos) += sizeof(double);
	return d;
}


static int lexportwriter(lua_State *L, const void* chunk, size_t size, void* data) 
{
	(void)L;
	buffer* b = (buffer*)data;
// 	encodeInteger(size, b);
	encodeBuffer(chunk, size, b);
	return 0;
}
// const char* limportreader(lua_State* L, void* data, size_t* size)
// {
// 	(void)L;
// 	buffer* b = (buffer*)data;
// 	*size = b->size;
// 	return b->buf;
// }


void _exportLuaVariable(lua_State* L, int index, buffer* b)
{
	int t = lua_type(L, index);
	int tablesize;
	const char* c;

	encodeInteger(t, b);
	
	switch(t)
	{
		case LUA_TNIL:
		break;
		case LUA_TBOOLEAN:
			encodeInteger(lua_toboolean(L, index), b);
		break;
		case LUA_TLIGHTUSERDATA:
			luaL_error(L, "Cannot export LIGHTUSERDATA");
		break;
		case LUA_TNUMBER:
			encodeDouble(lua_tonumber(L, index), b);
		break;
		case LUA_TSTRING:
			c = lua_tostring(L, index);
			tablesize = strlen(c) + 1;
			encodeInteger(tablesize, b);
			encodeBuffer(c, strlen(c)+1, b);
		break;
		case LUA_TTABLE:
			tablesize = 0;

			lua_pushnil( L );
			while(lua_next( L, index))
			{
				tablesize++;
				lua_pop( L, 1 );
			}
			lua_pop( L, 1 );
			
			encodeInteger(tablesize, b);

			lua_pushnil(L);
			while(lua_next(L, index) != 0)
			{
				_exportLuaVariable(L, -2, b);
				_exportLuaVariable(L, -1, b);
				lua_pop(L, 1);
			}
			lua_pop( L, 1 );
		break;
		case LUA_TFUNCTION:
		{
			buffer b2;
			b2.pos = 0;
			b2.size = 0;
			b2.buf = 0;
			
			lua_pushvalue(L, index); //copy func to top
			if(lua_dump(L, lexportwriter, &b2) != 0)
				luaL_error(L, "Unable to encode function");
			
			lua_pop(L, 1);
			
			if(b2.pos)
			{
				encodeInteger(b2.pos, b);
				encodeBuffer(b2.buf, b2.pos, b);
				free(b2.buf);
			}
		}
		break;
		case LUA_TUSERDATA:
			luaL_error(L, "Cannot export USERDATA");
		break;
		case LUA_TTHREAD:
			luaL_error(L, "Cannot export THREAD");
		break;
		default:
			luaL_error(L, "unknown type: %i", t);
	}
}

char* exportLuaVariable(lua_State* L, int index, int* chunksize)
{
	buffer b;
	b.buf  = (char*)malloc(32);
	b.size = 32;
	b.pos  = 0;
	
	_exportLuaVariable(L, index, &b);
	
	*chunksize = b.pos;
	return b.buf;
}

int _importLuaVariable(lua_State* L, char* chunk, int* pos, int chunksize)
{
	int t = decodeInteger(chunk, pos);
	int i;

	switch(t)
	{
		case LUA_TNIL:
			lua_pushnil(L);
		break;
		case LUA_TBOOLEAN:
			lua_pushboolean(L, decodeInteger(chunk, pos));
		break;
		case LUA_TLIGHTUSERDATA:
			luaL_error(L, "Cannot import LIGHTUSERDATA");
		break;
		case LUA_TNUMBER:
			lua_pushnumber(L, decodeDouble(chunk, pos));
		break;
		case LUA_TSTRING:
		{
			int len = decodeInteger(chunk, pos);
			char* s = (char*)malloc(len);
			memcpy(s, chunk + (*pos), len);
			(*pos) += len;
			
			lua_pushstring(L, s);
			free(s);
		}
		break;
		case LUA_TTABLE:
		{
			lua_newtable(L);
			int ts = decodeInteger(chunk, pos);
			
			for(int i=0; i<ts; i++)
			{
				_importLuaVariable(L, chunk, pos, chunksize);
				_importLuaVariable(L, chunk, pos, chunksize);
				lua_settable(L, -3);
			}
		}
		break;
		case LUA_TFUNCTION:
		{
			int chunksize = decodeInteger(chunk, pos);

			luaL_loadbuffer(L, chunk + *pos, chunksize, "import_function");
			*pos = *pos + chunksize;
// 			buffer b;
// 			b.buf = chunk + *pos;
// 			b.pos = 0;
// 			b.size = chunksize;
// 			
// 			if(lua_load(L, limportreader, &b, "import_function"))
// 				printf("%s\n", lua_tostring(L, -1));
		}
		break;
		case LUA_TUSERDATA:
			luaL_error(L, "Cannot import USERDATA");
		break;
		case LUA_TTHREAD:
			luaL_error(L, "Cannot import THREAD");
		break;
		default:
			luaL_error(L, "unknown import type: %i", t);
	}
}

int importLuaVariable(lua_State* L, char* chunk, int chunksize)
{
	int pos = 0;
	_importLuaVariable(L, chunk, &pos, chunksize);
}
