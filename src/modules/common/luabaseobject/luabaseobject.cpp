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

#include "luabaseobject.h"
#include <string.h>
#include <stdlib.h>

LuaBaseObject::LuaBaseObject(int t)
{
	type = t;
	refcount = 0;
	L = 0;
}

void LuaBaseObject::encode(buffer* b)
{
	fprintf(stderr, "Encode has not been written for `%s'\n", lineage(0));
}

int  LuaBaseObject::decode(buffer* b)
{
	fprintf(stderr, "Decode has not been written for `%s'\n", lineage(0));
	return 0;
}


static int l_lineage(lua_State* L)
{
	LUA_PREAMBLE(LuaBaseObject, b, 1);
	
	lua_newtable(L);
	for(int i=1; i<=5; i++)
	{
		const char* l = b->lineage(i-1);
		if(l)
		{
			lua_pushinteger(L, i);
			lua_pushstring(L, l);
			lua_settable(L, -3);
		}
	}
	return 1;
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* LuaBaseObject::luaMethods()
{
	if(m[127].name)return m;

	static const luaL_Reg _m[] =
	{
		{"lineage", l_lineage},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}


	
static int addReg(luaL_Reg* old_vals, const luaL_Reg* new_val)
{
	int i = 0;
	for(i=0; old_vals[i].name; i++)
	{
		if(strcmp(new_val->name, old_vals[i].name) == 0)
		//then we replace this one
		{
			old_vals[i].func = new_val->func;
			return 1;
		}
	}
	if(i < 127)
	{
		old_vals[i].func = new_val->func;
		old_vals[i].name = new_val->name;
		return 1;
	}
	return 0;
}

void merge_luaL_Reg(luaL_Reg* old_vals, const luaL_Reg* new_vals)
{
	if(!new_vals)return;
	for(int i=0; new_vals[i].name; i++)
	{
		if(!addReg(old_vals, &(new_vals[i])))
		{
			fprintf(stderr, "(%s:%i) metatable larger than arbitrary cuttoff (127)\n", __FILE__, __LINE__);
			// warning/error
		}
	}
}




static void ensureSize(int add, buffer* b)
{
	if(b->pos + add >= b->size)
	{
		b->size *= 2;
		if(b->pos + add >= b->size)
			b->size = b->pos + add + 1;

		b->buf = (char*)realloc(b->buf, b->size);
	}
}

void encodeBuffer(const void* s, const int len, buffer* b)
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

void encodeChar(const char c, buffer* b)
{
	encodeBuffer(&c, sizeof(char), b);
}


void decodeBuffer(void* dest, const int len, buffer* b)
{
	memcpy(dest, b->buf+b->pos, len);
	b->pos += len;
}
int decodeInteger(buffer* b)
{
	int i;
	decodeBuffer(&i, sizeof(int), b);
	return i;
}
double decodeDouble(buffer* b)
{
	double d;
	decodeBuffer(&d, sizeof(double), b);
	return d;
}
char decodeChar(buffer* b)
{
	char c;
	decodeBuffer(&c, sizeof(char), b);
	return c;
}



#include "info.h"
extern "C"
{
LUABASEOBJECT_API int lib_register(lua_State* L);
LUABASEOBJECT_API int lib_version(lua_State* L);
LUABASEOBJECT_API const char* lib_name(lua_State* L);
LUABASEOBJECT_API int lib_main(lua_State* L);
}

LUABASEOBJECT_API int lib_register(lua_State* L)
{
	return 0;
}

LUABASEOBJECT_API int lib_version(lua_State* L)
{
	return __revi;
}

const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "LuaBaseObject";
#else
	return "LuaBaseObject-Debug";
#endif
}

int lib_main(lua_State* L)
{
	return 0;
}

