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

//static FILE* report = stdout;
#define report 0

LuaBaseObject::LuaBaseObject(int t)
{
	type = t;
	refcount = 0;
	L = 0;
}

void LuaBaseObject::encode(buffer* /*b*/)
{
	fprintf(stderr, "Encode has not been written for `%s'\n", lineage(0));
}

int  LuaBaseObject::decode(buffer* /*b*/)
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

void merge_luaL_pair(luaL_Reg* old_vals, const char* name, lua_CFunction func)
{
    luaL_Reg r;
    r.name = name;
    r.func = func;

    if(!addReg(old_vals, &r))
    {
        fprintf(stderr, "(%s:%i) metatable larger than arbitrary cuttoff (127)\n", __FILE__, __LINE__);
        // warning/error
    }
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

static void _encodeBuffer(const void* s, const int len, buffer* b)
{
    ensureSize(len, b);
    memcpy(b->buf + b->pos, s, len);
    b->pos += len;
}


void encodeBuffer(const void* s, const int len, buffer* b)
{
    if(report)
	fprintf(report, "(%s:%04i) Encoding Buffer: %d bytes\n", __FILE__, __LINE__, len);
    _encodeBuffer(s, len, b);
}

void encodeDouble(const double d, buffer* b)
{
	// encodeBuffer("d", 1, b); // temp
    if(report)
	fprintf(report, "(%s:%04i) Encoding Double %g\n", __FILE__, __LINE__, d);
    _encodeBuffer(&d, sizeof(d), b);
}

void encodeInteger(const int i, buffer* b)
{
	// encodeBuffer("i", 1, b); // temp
    if(report)
	fprintf(report, "(%s:%04i) Encoding Integer %d\n", __FILE__, __LINE__, i);
    _encodeBuffer(&i, sizeof(i), b);
}

void encodeChar(const char c, buffer* b)
{
    if(report)
	fprintf(report, "(%s:%04i) Encoding Char %c (0x%X)\n", __FILE__, __LINE__, c, (int)c);
	// encodeBuffer("c", 1, b); // temp
    _encodeBuffer(&c, sizeof(char), b);
}

int  encodeContains(LuaBaseObject* o, buffer* b)
{
    for(int i=0; i< (int)b->encoded.size(); i++)
    {
	if(b->encoded[i] == o)
	{
	    if(report)
		fprintf(report, "(%s:%04i) Buffer contains object\n", __FILE__, __LINE__);
	    return 1;
	}
    }
    if(report)
	fprintf(report, "(%s:%04i) Buffer does not contain object\n", __FILE__, __LINE__);
    return 0;
}

void encodeOldThis (LuaBaseObject* o, buffer* b)
{
    int enc_idx = -1;
    for(int i=0; i<(int)b->encoded.size() && enc_idx < 0; i++)
    {
	if(b->encoded[i] == o)
	    enc_idx = i;
    }

    if(report)
    {
	fprintf(report, "(%s:%04i) encodeOldThis, idx = %i\n", __FILE__, __LINE__, enc_idx);
    }

    encodeChar(ENCODE_MAGIC_OLD, b);
    encodeInteger(enc_idx, b);
}

void encodeNewThis (LuaBaseObject* o, buffer* b)
{
    int enc_idx = b->encoded.size();

    if(report)
    {
	fprintf(report, "(%s:%04i) encodeNewThis, idx = %i\n", __FILE__, __LINE__, enc_idx);
    }
    // printf("new encoded. idx = %i\n", enc_idx);
    b->encoded.push_back(o);
    encodeChar(ENCODE_MAGIC_NEW, b);
    encodeInteger(enc_idx, b);
}





static void _decodeBuffer(void* dest, const int len, buffer* b)
{
    if(b->pos + len > b->size)
    {
	int* i = (int*)5;
	*i = 5;
    }
    memcpy(dest, b->buf+b->pos, len);
    b->pos += len;
}

void decodeBuffer(void* dest, const int len, buffer* b)
{
    if(report)
	fprintf(report, "(%s:%04i) Decoding Buffer: %d bytes\n", __FILE__, __LINE__, len);
    _decodeBuffer(dest, len, b);
}


static int check_type(const char* t, buffer* b)
{
	char d; decodeBuffer(&d, 1, b);
	if(d != t[0])
	{
		fprintf(stderr, "Stream data mismatch. Expected %c, read %c. Crashing.\n", t[0], d);
		int* q = (int*)5;
		*q = 5;
	}
}

int decodeInteger(buffer* b)
{
    // check_type("i", b);
    int i;
    _decodeBuffer(&i, sizeof(int), b);
    if(report)
	fprintf(report, "(%s:%04i) Decoding Integer %d\n", __FILE__, __LINE__, i);
    
    return i;
}
double decodeDouble(buffer* b)
{
    // check_type("d", b);
    double d;
    _decodeBuffer(&d, sizeof(double), b);
    if(report)
	fprintf(report, "(%s:%04i) Decoding Double %g\n", __FILE__, __LINE__, d);
    return d;
}
char decodeChar(buffer* b)
{
    // check_type("c", b);
    char c;
    _decodeBuffer(&c, sizeof(char), b);
    if(report)
	fprintf(report, "(%s:%04i) Decoding Char %c (0x%02X)\n", __FILE__, __LINE__, c, (int)c);
    return c;
}


/*
template<class T>
void encodeT(T* lbo, buffer* b)
{
    encodeInteger(LUA_TUSERDATA, b);
    encodeInteger(lbo->type, b);
    lbo->encode(b);
}

template<class T>
T* decodeT(lua_State* L, buffer* b)
{
	return dynamic_cast<T*>(decodeLuaBaseObject(L, b));
}
*/

LuaBaseObject* decodeLuaBaseObject(lua_State* L, buffer* b)
{
	int TUSERDATA = decodeInteger(b);
	if(TUSERDATA != LUA_TUSERDATA)
	{
		fprintf(stderr, "(%s:%i) Expectd LUA_TUSERDATA(%i) in stream, got %i\n", __FILE__, __LINE__, LUA_TUSERDATA, TUSERDATA);
		int* q = (int*)5;
		*q = 5; // force traceable crash
	}

	int type = decodeInteger(b);
	char magic = decodeChar(b);
	int pos = decodeInteger(b);

	if(report)
	    fprintf(report, "(%s:%04i) Decoding `%s' (type=%d)\n", __FILE__, __LINE__, Factory_typename(type), type);

	if((magic != ENCODE_MAGIC_NEW) && (magic != ENCODE_MAGIC_OLD))
	{
		fprintf(stderr, "(%s:%i)Malformed data stream:\n", __FILE__, __LINE__);
		fprintf(stderr, "(%s:%i)Expected New or Old flag (%i or %i), got %i\n",  __FILE__, __LINE__, ENCODE_MAGIC_NEW, ENCODE_MAGIC_OLD, magic);
		fprintf(stderr, "(%s:%i)Type, Magic, Pos = %i, %i, %i\n", __FILE__, __LINE__, type, magic, pos);
		return 0;
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
		}
		else
		{
			fprintf(stderr, "Failed to create new type from factory\n");
		}

	}
	if(magic == ENCODE_MAGIC_OLD)
	{
	    if(report)
		fprintf(report, "(%s:%04i) Looking up old object\n", __FILE__, __LINE__);

		if(pos < 0 || pos >= (int)b->encoded.size())
		{
			fprintf(stderr, "(%s:%i)Malformed data stream\n", __FILE__, __LINE__);
		}
		else
		{
			e->L = L;
			e = (LuaBaseObject*)b->encoded[pos];
		}
	}
	//luaT_inc<LuaBaseObject>(e);
	return e;
}


#ifdef _CREATE_LIBRARY

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

#endif
