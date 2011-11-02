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

extern "C" {
	#include <lua.h>
	#include <lualib.h>
	#include <lauxlib.h>
}

#include "encodable.h"

#ifdef WIN32
#ifdef ENCODE_EXPORTS
ENCODE_API char* exportLuaVariable(lua_State* L, int index,   int* chunksize);
ENCODE_API int   importLuaVariable(lua_State* L, char* chunk, int  chunksize);

ENCODE_API void _exportLuaVariable(lua_State* L, int index, buffer* b);
ENCODE_API int _importLuaVariable(lua_State* L, buffer* b);
#else
#include <windows.h>
static char* exportLuaVariable(lua_State* L, int index, int* chunksize)
{
	typedef char* (*func)(lua_State*, int, int*); 
	static func thefunc = 0;

	if(!thefunc)
	{
		// first need to get function to get lib paths
		typedef const char* (*sfuncs) (const char*);
		sfuncs getPath = import_function<sfuncs>("", "get_libpath");

		if(!getPath)
		{
			fprintf(stderr, "Failed to load `get_libpath'\n");
			return 0;
		}

		const char* encPath = getPath("encode");

		if(!encPath)
			return 0;

		thefunc = import_function<func>(encPath, "exportLuaVariable");
	}

	if(!thefunc)
	{
		printf("failed to load exportLuaVariable\n");
		return NULL;
	}
	return thefunc(L, index, chunksize);
}

static int   importLuaVariable(lua_State* L, char* chunk, int  chunksize)
{
	typedef int (*func)(lua_State*, char*, int);
	static func thefunc = 0;

	if(!thefunc)
	{
		// first need to get function to get lib paths
		typedef const char* (*sfuncs) (const char*);
		sfuncs getPath = import_function<sfuncs>("", "get_libpath");

		if(!getPath)
		{
			fprintf(stderr, "Failed to load `get_libpath'\n");
			return 0;
		}

		const char* encPath = getPath("encode");

		if(!encPath)
			return 0;

		thefunc = import_function<func>(encPath, "importLuaVariable");
	}

	if(!thefunc)
	{
		printf("failed to load Factory_newItem\n");
		return NULL;
	}
	return thefunc(L, chunk, chunksize);
}



static void _exportLuaVariable(lua_State* L, int index, buffer* b)
{
	typedef void(*func)(lua_State*, buffer*); 
	static func thefunc = 0;

	if(!thefunc)
	{
		// first need to get function to get lib paths
		typedef const char* (*sfuncs) (const char*);
		sfuncs getPath = import_function<sfuncs>("", "get_libpath");

		if(!getPath)
		{
			fprintf(stderr, "Failed to load `get_libpath'\n");
			return;
		}

		const char* encPath = getPath("encode");

		if(!encPath)
			return;

		thefunc = import_function<func>(encPath, "_exportLuaVariable");
	}

	if(!thefunc)
	{
		printf("failed to load _exportLuaVariable\n");
		return;
	}
	thefunc(L, b);
}


static int _importLuaVariable(lua_State* L, buffer* b)
{
	typedef int (*func)(lua_State*, buffer*);
	static func thefunc = 0;

	if(!thefunc)
	{
		// first need to get function to get lib paths
		typedef const char* (*sfuncs) (const char*);
		sfuncs getPath = import_function<sfuncs>("", "get_libpath");

		if(!getPath)
		{
			fprintf(stderr, "Failed to load `get_libpath'\n");
			return 0;
		}

		const char* encPath = getPath("encode");

		if(!encPath)
			return 0;

		thefunc = import_function<func>(encPath, "_importLuaVariable");
	}

	if(!thefunc)
	{
		printf("failed to load _importLuaVariable\n");
		return NULL;
	}
	return thefunc(L, b);
}






#endif
#else
ENCODE_API char* exportLuaVariable(lua_State* L, int index,   int* chunksize);
ENCODE_API int   importLuaVariable(lua_State* L, char* chunk, int  chunksize);

ENCODE_API void _exportLuaVariable(lua_State* L, int index, buffer* b);
ENCODE_API int _importLuaVariable(lua_State* L, buffer* b);
#endif
