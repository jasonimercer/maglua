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

#ifndef LUABASEOBJECT_H
#define LUABASEOBJECT_H
#include "factory.h"


#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)

 #ifdef LUABASEOBJECT_EXPORTS
  #define LUABASEOBJECT_API __declspec(dllexport)
 #else
  #define LUABASEOBJECT_API __declspec(dllimport)
 #endif
#else
 #define LUABASEOBJECT_API 
#endif



LUABASEOBJECT_API typedef struct buffer
{
	char* buf;
	int pos;
	int size;
}buffer;

#define ENCODE_UNKNOWN      0

#define ENCODE_SPINSYSTEM   hash32("SpinSystem")
#define ENCODE_ANISOTROPY   hash32("Anisotropy")
#define ENCODE_APPLIEDFIELD hash32("AppliedField")
#define ENCODE_DIPOLE       hash32("Dipole")
#define ENCODE_EXCHANGE     hash32("Exchange")
#define ENCODE_THERMAL      hash32("Thermal")

#define ENCODE_LLGCART      hash32("LLGCart")
#define ENCODE_LLGQUAT      hash32("LLGQuat")
#define ENCODE_LLGFAKE      hash32("LLGFake")
#define ENCODE_LLGALIGN     hash32("LLGALign")

#define ENCODE_INTERP2D    hash32("Interpolate2D")
#define ENCODE_INTERP1D    hash32("interpolate1D")
#define ENCODE_MAGNETOSTATIC hash32("Magnetostatic")

#define ENCODE_SHORTRANGE  hash32("ShortRange")

extern "C"
{
LUABASEOBJECT_API   void encodeBuffer(const void* s, const int len, buffer* b);
LUABASEOBJECT_API   void encodeDouble(const double d, buffer* b);
LUABASEOBJECT_API   void encodeInteger(const int i, buffer* b);
LUABASEOBJECT_API    int decodeInteger(buffer* b);
LUABASEOBJECT_API double decodeDouble(buffer* b);
LUABASEOBJECT_API   void decodeBuffer(void* dest, const int len, buffer* b);
}

#include <string.h>
#include <string>
#include <vector>
using namespace std;

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}

#define LINEAGE5(v1,v2,v3,v4,v5) \
	virtual const char* lineage(int i) { switch(i) { \
	case 0: return v1; case 1: return v2; \
	case 2: return v3; case 3: return v4; \
	case 4: return v5; } return 0; } \
	static const char* slineage(int i) { switch(i) { \
	case 0: return v1; case 1: return v2; \
	case 2: return v3; case 3: return v4; \
	case 4: return v5; } return 0; } \
	static const char* typeName() {return v1;}

#define LINEAGE4(v1,v2,v3,v4) LINEAGE5(v1,v2,v3,v4, 0)
#define LINEAGE3(v1,v2,v3)    LINEAGE4(v1,v2,v3, 0)
#define LINEAGE2(v1,v2)       LINEAGE3(v1,v2, 0)
#define LINEAGE1(v1)          LINEAGE2(v1, 0)
#define LINEAGE0()            LINEAGE1(0)

class LUABASEOBJECT_API LuaBaseObject
{
public:
	LuaBaseObject(int type = 0);

	LINEAGE1("LuaBaseObject")
	static luaL_Reg* luaMethods() {return 0;}
	virtual int luaInit(lua_State* /*L*/) {return 0;}
	virtual void push(lua_State* /*L*/) {}
	static int help(lua_State* /*L*/) {return 0;}

	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);
	
	//string name;
	int type;
	int refcount;
	lua_State* L;
};


// this is the common preamble for most
// lua methods: Check to see if the element
// at the index is of the correct type
// and return 0 if it's not.
// T Class, t variable name, i stack index
#define LUA_PREAMBLE(T,t,i) \
	T* t = luaT_to<T>(L, i); \
	if(!t) return 0;

// decrement refcounter. delete if needed.
// always return resulting pointer
// does not rely on a lua_State
template<class T>
T* luaT_dec(T* t)
{
	if(!t)
		return 0;
	t->refcount--;
	if(t->refcount == 0)
	{
		delete t;
		return 0;
	}

	return t;
}

// increment refcounter. This could easily
// be a method of the baseClass but decrement
// cannot without some sort of a delayed delete
// mechanism. Keeping increment as an external
// function for symmetry
template<class T>
T* luaT_inc(T* t)
{
	if(!t)
		return 0;
	t->refcount++;
	return t;
}

// test type
template<class T>
int luaT_is(lua_State* L, int idx)
{
	if(!lua_isuserdata(L, idx))
	{
		return 0;
	}

	LuaBaseObject** pp = (LuaBaseObject**)lua_touserdata(L, idx);

	if(!pp)
		return 0;

	int eq = 0;
	int i=0;
	while((*pp)->lineage(i) && !eq)
	{
		if(strcmp(T::typeName(), (*pp)->lineage(i)) == 0)
			eq = 1;
		i++;
	}

	return eq;
}

// convert an object on a stack to a C pointer
template<class T>
T* luaT_to(lua_State* L, int idx)
{
	if(luaT_is<T>(L, idx))
	{
		T** pp = (T**)lua_touserdata(L, idx);
		if(pp)
			return *pp;
		printf("null!\n");
	}

	char msg[128];
	snprintf(msg, 128, "%s expected, got %s", T::typeName(), lua_typename(L, lua_type(L, idx)));
	luaL_argerror(L, idx, msg);
	return 0;
}

// push an object on a Lua stack
template<class T>
void luaT_push(lua_State* L, LuaBaseObject* tt)
{
	T* t = dynamic_cast<T*>(tt);
	if(!t)
	{
		lua_pushnil(L);
	}
	else
	{
		T** pp = (T**)lua_newuserdata(L, sizeof(T**));
		*pp = luaT_inc<T>(t);

		luaL_getmetatable(L, T::typeName());
		lua_setmetatable(L, -2);
	}
}

// garbage collection metamethod.
// decrement refcount, delete if needed
template<class T>
int luaT_gc(lua_State* L)
{
	luaT_dec<T>(luaT_to<T>(L, 1));

	return 0;
}

// push metatable on the stack
template<class T>
int luaT_mt(lua_State* L)
{
	luaL_getmetatable(L, T::typeName());
	return 1;
}

// basic tostring metamathod acting like type()
template<class T>
int luaT_tostring(lua_State* L)
{
	T* t = luaT_to<T>(L, 1);
	if(!t) return 0;

	lua_pushfstring(L, "%s %p", T::typeName(), t);
	//lua_pushfstring(L, "%s", T::typeName());
	return 1;
}

template<class T>
int luaT_setname(lua_State* L)
{
	T* t = luaT_to<T>(L, 1);
	if(!t) return 0;

	t->name = lua_tostring(L, 2);
	return 0;
}

template<class T>
int luaT_getname(lua_State* L)
{
	T* t = luaT_to<T>(L, 1);
	if(!t) return 0;

	lua_pushstring(L, t->name.toStdString().c_str());
	return 1;
}

// create a new object and push it on the stack
template<class T>
int luaT_new(lua_State* L)
{
	T* t = new T;
	t->luaInit(L);
	luaT_push<T>(L, t);
	return 1;
}

template<class T>
int luaT_help(lua_State* L)
{
	return T::help(L);
}

// add methods to the metamethod table of an object type
template<class T>
void luaT_addMethods(lua_State* L, const luaL_Reg* methods)
{
	if(!methods)
		return;
	luaL_getmetatable(L, T::typeName());
	luaL_register(L, NULL, methods);
	lua_pop(L,1);
}

template<class T>
LuaBaseObject* new_luabaseobject()
{
	return new T;
}

template<class T>
inline void luaT_register(lua_State* L)
{
	const int top = lua_gettop(L);
	if(!luaL_newmetatable(L, T::typeName()))
		return;
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);
	lua_settable(L, -3);
	lua_pushstring(L, "__gc");
	lua_pushcfunction(L, luaT_gc<T>);
	lua_settable(L, -3);
	lua_pushstring(L, "__tostring");
	lua_pushcfunction(L, luaT_tostring<T>);
	lua_settable(L, -3);
	
	vector<string> list;
	const char* tt = T::typeName();
	for(int i=0; tt && tt[i];)
	{
		string buffer;
		if(tt[i] == '.') i++;
		for(;tt[i] && tt[i] != '.'; i++)
			buffer.push_back(tt[i]);
		
		if(buffer.length() > 0)
			list.push_back(buffer);
	}
	
	if(list.size() > 1)
	{
		lua_getglobal(L, list[0].c_str());
		if(lua_isnil(L, -1))
		{
			lua_pop(L, 1);
			lua_newtable(L);
			lua_setglobal(L, list[0].c_str());
			lua_getglobal(L, list[0].c_str());
		}
		for(unsigned int i=1; i<list.size()-1; i++)
		{
			lua_getfield(L, -1, list[i].c_str());
			if(lua_isnil(L, -1))
			{
				lua_pop(L, 1);
				lua_newtable(L);
				lua_setfield(L, -2, list[i].c_str());
				lua_getfield(L, -1, list[i].c_str());
			}
		}
	}

	lua_newtable(L);
	lua_pushstring(L, "new");
	lua_pushcfunction(L, luaT_new<T>);
	lua_settable(L, -3);
	lua_pushstring(L, "metatable");
	lua_pushcfunction(L, luaT_mt<T>);
	lua_settable(L, -3);
	lua_pushstring(L, "help");
	lua_pushcfunction(L, luaT_help<T>);
	lua_settable(L, -3);

	if(list.size() > 1)
	{
		lua_setfield(L, -2, list.back().c_str());
	}
	else
	{
		lua_setglobal(L, T::typeName());
	}


	if(T::luaMethods())
		luaT_addMethods<T>(L, T::luaMethods());

	Factory_registerItem(hash32(T::typeName()), new_luabaseobject<T>, luaT_push<T>, T::typeName());
	
	while(lua_gettop(L) > top)
		lua_pop(L, 1);
}

#define _NULLPAIR1    {NULL, NULL}
#define _NULLPAIR2   _NULLPAIR1, _NULLPAIR1
#define _NULLPAIR4   _NULLPAIR2, _NULLPAIR2
#define _NULLPAIR8   _NULLPAIR4, _NULLPAIR4
#define _NULLPAIR16  _NULLPAIR8, _NULLPAIR8
#define _NULLPAIR32  _NULLPAIR16,_NULLPAIR16
#define _NULLPAIR64  _NULLPAIR32,_NULLPAIR32
#define _NULLPAIR128 _NULLPAIR64,_NULLPAIR64

void merge_luaL_Reg(luaL_Reg* old_vals, const luaL_Reg* new_vals);


// macros to create simple getter/setter functions
#define LUAFUNC_SET_DOUBLE(T,var,func_name) \
static int func_name(lua_State* L) \
{ \
	LUA_PREAMBLE(T, _x_, 1); \
	(_x_)->var = lua_tonumber(L, 2); \
	return 0; \
}

#define LUAFUNC_GET_DOUBLE(T,var,func_name) \
static int func_name(lua_State* L) \
{ \
	LUA_PREAMBLE(T, _x_, 1); \
	lua_pushnumber(L, (_x_)->var); \
	return 1; \
}


#endif // LUABASEOBJECT_H
