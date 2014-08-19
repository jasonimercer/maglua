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


//static FILE* report = stdout;
#define report 0


const char* find_cfunc_code = 
    "return function(target)\n"
    "  for k1,v1 in pairs(_G) do\n"
    "    if v1 == target then return k1 end\n"
    "    if type(v1) == type({}) and k1 ~= \"_G\" then\n"
    "      for k2,v2 in pairs(v1) do\n"
    "        if v2 == target then return k1..\".\"..k2 end\n"
    "        if type(v2) == type({}) and k2 ~= \"_G\" then\n"
    "          for k3,v3 in pairs(v2) do\n"
    "            if v3 == target then return k1..\".\"..k2..\".\"..k3 end\n"
    "          end\n"
    "        end\n"
    "      end\n"
    "    end\n"
    "  end\n"
    "end\n";


#ifdef size64_t
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

	if(b->debug)
	{
		fprintf(b->debug, "(%s:%i) exporting %12s pos:%6i\n", __FILE__, __LINE__, lua_typename(L, t), b->pos);
	}
	
	if(report)
	    fprintf(report, "(%s:%04i) Encoding `%s' (lua_type=%i)\n", __FILE__, __LINE__, lua_typename(L, t), t);

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
		{
			// printf("exporting table\n");
			const void* tab_ptr = lua_topointer(L, index);
			int existing_index = -1;
			for(unsigned int i=0; i<b->encoded_table_pointers.size() &&
				existing_index == -1; i++)
			{
				if(tab_ptr == b->encoded_table_pointers[i])
					existing_index = i;
			}

			if(existing_index == -1) //new
			{
			    if(report)
			    {
				fprintf(report, "(%s:%04i) Buffer does not contain table\n", __FILE__, __LINE__);
				fprintf(report, "(%s:%04i) New table index: %d\n", __FILE__, __LINE__, (int)b->encoded_table_pointers.size());
			    }

			    b->encoded_table_pointers.push_back(tab_ptr);
			    encodeChar(ENCODE_MAGIC_NEW, b);
			    encodeInteger(0, b); // for NEW/OLD symmetry
			    
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
				    if(report)
				    {
					lua_getglobal(L, "tostring");
					lua_pushvalue(L, -3);
					lua_pcall(L, 1,1,0);
					lua_getglobal(L, "tostring");
					lua_pushvalue(L, -3);
					lua_pcall(L, 1,1,0);
					fprintf(report, "(%s:%04i) Exporting Key/Value %s/%s:\n", __FILE__, __LINE__, lua_tostring(L, -2), lua_tostring(L, -1));
					lua_pop(L, 2);
				    }
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
			}
			else
			{
			    if(report)
			    {
				fprintf(report, "(%s:%04i) Buffer does contain table\n", __FILE__, __LINE__);
				fprintf(report, "(%s:%04i) Table index: %i\n", __FILE__, __LINE__, existing_index);
			    }
			    encodeChar(ENCODE_MAGIC_OLD, b);
			    encodeInteger(existing_index, b);
			}
			break;
		}
		case LUA_TFUNCTION:
		{
		    int c = lua_iscfunction(L, index);
		    encodeInteger(c, b);

		    if(c)
		    {
			// can't encode a C function so we'll iterate through the tree and look
			// for a match and encode the name. Inefficient but it'll work for now
			lua_CFunction func = lua_tocfunction(L, index);


			if(luaL_loadstring(L, find_cfunc_code))
			{
 			    fprintf(stderr, "(%s:%04i)%s\n", __FILE__, __LINE__, lua_tostring(L, -1));
			}

			if(lua_pcall(L, 0,1,0))
			{
 			    fprintf(stderr, "(%s:%04i)%s\n", __FILE__, __LINE__, lua_tostring(L, -1));
			}

			// now the above function is on the stack so we can use it to find our target
			lua_pushvalue(L, index); //copy func to top
			
			if(lua_pcall(L, 1,1,0))
			{
 			    fprintf(stderr, "(%s:%04i)%s\n", __FILE__, __LINE__, lua_tostring(L, -1));
			}


			if(lua_isstring(L, -1))
			{
			    const char* func_name = lua_tostring(L, -1);
			    int len = strlen(func_name) + 1;
			    encodeInteger(len, b);
			    encodeBuffer(func_name, len, b);
			    lua_pop(L, 1); // remove name
			}
			else
			{
			    encodeInteger(0, b); // failed to find the function. should tell someone
			}
		    }
		    else
		    {
			buffer* b2 = new buffer;
			b2->pos = 0;
			b2->size = 0;
			b2->buf = 0;
			
			lua_pushvalue(L, index); //copy func to top
			if(lua_dump(L, lexportwriter, b2) != 0)
				luaL_error(L, "Unable to encode function");
			
			lua_pop(L, 1);

			#if 0
			if(b->debug)
			{
				fprintf(b->debug, "(%s:%04i) function size: %i\n", __FILE__, __LINE__, b2->pos);
			}
			#endif
			
			if(b2->pos)
			{
				encodeInteger(b2->pos, b);
				encodeBuffer(b2->buf, b2->pos, b);
				free(b2->buf);
			}
			delete b2;
		    }

		    // now it's time for upvalues
		    int num_upvalues = 0;
		    for(int q=1; q<60; q++) // assuming less than 60 upvalues
		    {
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
			LuaBaseObject** pe = (LuaBaseObject**)lua_topointer(L, index);
			LuaBaseObject* e = *pe;
			if(report)
			{
                            fprintf(report, "(%s:%04i) Exporting `%s' (type=%i)\n", __FILE__, __LINE__, Factory_typename(e->type), e->type);
			}
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
			char magic = decodeChar(b);
			int pos = decodeInteger(b);

			if((magic != ENCODE_MAGIC_NEW) && (magic != ENCODE_MAGIC_OLD))
			{
				luaL_error(L, "(%s:%04i)Malformed data stream\n", __FILE__, __LINE__);
				return -1;
			}

			if(magic == ENCODE_MAGIC_NEW)
			{
				lua_newtable(L);
				int ref = luaL_ref(L, LUA_REGISTRYINDEX);
				lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
				b->encoded_table_refs.push_back(ref);

				int ts = decodeInteger(b); // table size


				for(int i=0; i<ts; i++)
				{
					_importLuaVariable(L, b);
					_importLuaVariable(L, b);
					if(report)
					{
					    lua_getglobal(L, "tostring");
					    lua_pushvalue(L, -3);
					    lua_pcall(L, 1,1,0);
					    lua_getglobal(L, "tostring");
					    lua_pushvalue(L, -3);
					    lua_pcall(L, 1,1,0);
					    fprintf(report, "(%s:%04i) Importing Key/Value %s/%s:\n", __FILE__, __LINE__, lua_tostring(L, -2), lua_tostring(L, -1));
					    lua_pop(L, 2);
					}
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
			if(magic == ENCODE_MAGIC_OLD)
			{
				if(pos < 0 || pos >= (int)b->encoded_table_refs.size())
				{
					luaL_error(L, "(%s:%04i)Malformed data stream\n", __FILE__, __LINE__);
				}
				else
				{
					int ref = b->encoded_table_refs[pos];
					lua_rawgeti(L, LUA_REGISTRYINDEX, ref);
				}
			}
		}
		break;
		case LUA_TFUNCTION:
		{
		    int cfunc = decodeInteger(b);
		    if(cfunc)
		    {
                        int len = decodeInteger(b);
			if(len == 0)
			{
			    lua_pushnil(L);
			}
			else
			{
			    char* s = (char*)malloc(len+15); // + 15 for "return    "  
			    snprintf(s, 15, "return    ");
			    memcpy(s+8, b->buf + b->pos, len);
			    b->pos += len;
			    
			    if(luaL_loadstring(L, s))
			    {
				fprintf(stderr, "(%s:%04i)%s\n", __FILE__, __LINE__, lua_tostring(L, -1));
			    }
			    
			    if(lua_pcall(L, 0,1,0))
			    {
				fprintf(stderr, "(%s:%04i)%s\n", __FILE__, __LINE__, lua_tostring(L, -1));
				fprintf(stderr, "Current state does not have `%s'\n", s+8);
			    }

			}
			
		    }
		    else
		    {
			int chunksize = decodeInteger(b);
			luaL_loadbuffer(L, b->buf + b->pos, chunksize, "import_function");
			b->pos += chunksize;
		    }
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
			if(report)
			    fprintf(report, "(%s:%04i) Decoding `%s'\n", __FILE__, __LINE__, Factory_typename(type));

			char magic = decodeChar(b);
			int pos = decodeInteger(b);

			if((magic != ENCODE_MAGIC_NEW) && (magic != ENCODE_MAGIC_OLD))
			{
				luaL_error(L, "(%s:%04i)Malformed data stream\n", __FILE__, __LINE__);
				return -1;
			}



			LuaBaseObject* e = 0;
			if(magic == ENCODE_MAGIC_NEW)
			{
			    if(report)
				fprintf(report, "(%s:%04i) Decoding new object\n", __FILE__, __LINE__);

				e = Factory_newItem(type);
				b->encoded.push_back(e);

				if(pos != (int)b->encoded.size()-1)
				    fprintf(stderr, "(%s:%i) Type=`%s'. Encoded index mismatch. Expected %i, got %i\n", __FILE__, __LINE__, Factory_typename(type),  pos, (int)b->encoded.size()-1);

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


			}
			if(magic == ENCODE_MAGIC_OLD)
			{
			    if(report)
				fprintf(report, "(%s:%04i) Looking up old object\n", __FILE__, __LINE__);


				if(pos < 0 || pos >= (int)b->encoded.size())
				{
					luaL_error(L, "(%s:%04i)Malformed data stream\n", __FILE__, __LINE__);
				}
				else
				{
					e = (LuaBaseObject*)b->encoded[pos];
					Factory_lua_pushItem(L, e, type);
				}
			}
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

	buffer_unref(L, &b);

	return 0;
}
