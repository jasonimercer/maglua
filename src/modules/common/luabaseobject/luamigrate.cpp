/******************************************************************************
* Copyright (C) 2008-2011 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "factory.h"
#include "luamigrate.h"
#include "luabaseobject.h"

#ifdef WIN32
//using size64_t (defined in luaconf.h) to allow compat with x86_64 bit linux
static int lexportwriter(lua_State *L, const void* chunk, size64_t size, void* data) 
#else
static int lexportwriter(lua_State *L, const void* chunk, size_t size, void* data) 
#endif
{
	(void)L;
	buffer* b = (buffer*)data;
	encodeBuffer(chunk, (const int)size, b);
	return 0;
}

void _exportLuaVariable(lua_State* L, int index, buffer* b)
{
	int t = lua_type(L, index);
	int tablesize;
	const char* c;

	if(index < 0)
	{
		index = lua_gettop(L) + index + 1;
	}

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
			// printf("exporting table\n");
			tablesize = 0;

			lua_pushnil( L );
			while(lua_next( L, index))
			{
				tablesize++;
				lua_pop( L, 1 );
			}
			
			encodeInteger(tablesize, b);

			lua_pushnil(L);
			while(lua_next(L, index) != 0)
			{
				_exportLuaVariable(L, -2, b);
				_exportLuaVariable(L, -1, b);
				lua_pop(L, 1);
			}

			if(	lua_getmetatable(L, index) == 0)
			{
				//printf("table does not have a metatable, pushing nil\n");
				lua_pushnil(L);
			}
			//printf("exporting metatable\n");
			_exportLuaVariable(L, lua_gettop(L), b);
			lua_pop(L, 1);
		break;
		case LUA_TFUNCTION:
		{
			buffer* b2 = new buffer;
			b2->pos = 0;
			b2->size = 0;
			b2->buf = 0;
			
			lua_pushvalue(L, index); //copy func to top
			if(lua_dump(L, lexportwriter, b2) != 0)
				luaL_error(L, "Unable to encode function");
			
			lua_pop(L, 1);
			
			if(b2->pos)
			{
				encodeInteger(b2->pos, b);
				encodeBuffer(b2->buf, b2->pos, b);
				free(b2->buf);
			}
			delete b2;

			// now it's time for upvalues
			int num_upvalues = 0;
			for(int q=1; q<60; q++) // assuming less than 60 upvalues
			{
				const char *lua_getupvalue (lua_State *L, int funcindex, int n);
				if( lua_getupvalue(L, index, q))
				{
					num_upvalues++;
					lua_pop(L, 1);
				}
			}

			encodeInteger(num_upvalues, b);

			for(int q=1; q<=num_upvalues; q++)
			{
				lua_getupvalue(L, index, q);
				_exportLuaVariable(L, lua_gettop(L), b);
			}
	   		lua_pop(L,num_upvalues);
		}
		break;
		case LUA_TUSERDATA:
		{
			//luaL_error(L, "Cannot export USERDATA");
			LuaBaseObject** pe = (LuaBaseObject**)lua_topointer(L, index);
			LuaBaseObject* e = *pe;
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
	b.buf  = 0;//(char*)malloc(32);
	b.size = 0;
	b.pos  = 0;
	
	_exportLuaVariable(L, index, &b);
	
	*chunksize = b.pos;
	return b.buf;
}

int _importLuaVariable(lua_State* L, buffer* b)
{
	int t = decodeInteger(b);

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

			_importLuaVariable(L, b); //import potential metatable
			if(lua_isnil(L, -1))
			{
				// no metatable
				lua_pop(L, 1);
			}
			else
			{
				lua_setmetatable(L, -2);
			}

		}
		break;
		case LUA_TFUNCTION:
		{
			int chunksize = decodeInteger(b);
			luaL_loadbuffer(L, b->buf + b->pos, chunksize, "import_function");
			b->pos += chunksize;
			int func_pos = lua_gettop(L);
			int num_upvalues = decodeInteger(b);
			for(int q=1; q<=num_upvalues; q++)
			{
				_importLuaVariable(L, b);
				lua_setupvalue(L, func_pos, q);
			}
		}
		break;
		case LUA_TUSERDATA:
		{
			int type = decodeInteger(b);
// 			printf(">>> %i, %i\n", lua_gettop(L), type);
			LuaBaseObject* e = Factory_newItem(type);
			if(e)
			{
				e->L = L;
				e->decode(b);
				Factory_lua_pushItem(L, e, type);
			}
			else
			{
				luaL_error(L, "Failed to create new type from factory\n");
			}
// 			printf(">>> %i\n", lua_gettop(L));
		}

		break;
		case LUA_TTHREAD:
			luaL_error(L, "Cannot import THREAD");
		break;
		default:
		{
			luaL_error(L, "unknown import type: %i", t);
		}
	}

	return 0;
}

int importLuaVariable(lua_State* L, char* chunk, int chunksize)
{
	buffer b;
	b.buf = chunk;
	b.pos = 0;
	b.size = chunksize;
	_importLuaVariable(L, &b);
	return 0;
}
