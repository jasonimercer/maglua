#include "luamigrate.h"
#include <string.h>
#include <stdlib.h>

#include "encodable.h"
#include "spinsystem.h"


static int lexportwriter(lua_State *L, const void* chunk, size_t size, void* data) 
{
	(void)L;
	buffer* b = (buffer*)data;
	encodeBuffer(chunk, size, b);
	return 0;
}

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
// 		case LUA_TLIGHTUSERDATA:
// 			luaL_error(L, "Cannot export LIGHTUSERDATA");
// 		break;
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
		{
			//luaL_error(L, "Cannot export USERDATA");
			const Encodable** pe = (const Encodable**)lua_topointer(L, index);
			const Encodable* e = *pe;
			encodeInteger(e->type, b);
			e->encode(b);
		}
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

int _importLuaVariable(lua_State* L, buffer* b)
{
	int t = decodeInteger(b);
	int i;

	switch(t)
	{
		case LUA_TNIL:
			lua_pushnil(L);
		break;
		case LUA_TBOOLEAN:
			lua_pushboolean(L, decodeInteger(b));
		break;
		case LUA_TLIGHTUSERDATA:
			luaL_error(L, "Cannot import LIGHTUSERDATA");
		break;
		case LUA_TNUMBER:
			lua_pushnumber(L, decodeDouble(b));
		break;
		case LUA_TSTRING:
		{
			int len = decodeInteger(b);
			char* s = (char*)malloc(len);
			memcpy(s, b->buf + b->pos, len);
			b->pos += len;
			lua_pushstring(L, s);
			free(s);
		}
		break;
		case LUA_TTABLE:
		{
			lua_newtable(L);
			int ts = decodeInteger(b);
			
			for(int i=0; i<ts; i++)
			{
				_importLuaVariable(L, b);
				_importLuaVariable(L, b);
				lua_settable(L, -3);
			}
		}
		break;
		case LUA_TFUNCTION:
		{
			int chunksize = decodeInteger(b);
			luaL_loadbuffer(L, b->buf + b->pos, chunksize, "import_function");
			b->pos += chunksize;
		}
		break;
		case LUA_TUSERDATA:
		{
			int type = decodeInteger(b);
			switch(type)
			{
				case ENCODE_SPINSYSTEM:
				{
					SpinSystem* ss = new SpinSystem(2,2,2);
					ss->decode(b);
					lua_pushSpinSystem(L, ss);
				}
				break;
				
				//default: //TYPE_NOEXPORT
			}


			
		}
// 		break;
// 			luaL_error(L, "Cannot import USERDATA");
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
	buffer b;
	b.buf = chunk;
	b.pos = 0;
	b.size = chunksize;
	_importLuaVariable(L, &b);
}
