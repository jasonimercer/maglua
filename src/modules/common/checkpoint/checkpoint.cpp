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

// 
// CheckPoint adds 2 functions.
// 
// checkpointSave(filename, a, b, c, d, ...)
//  saves the values a, b, c, d ... to the file: filename
// 
// a, b, c, d, ... = checkpointLoad(fn)
//  loads the variables in the file: filename to the 
//  variables a, b, c, d, ...

#include "luamigrate.h"
#include "checkpoint.h"
#include <stdio.h>
#include <stdlib.h>
#include "info.h"

static int sure_fwrite(const void* _data, int sizeelement, int numelement, FILE* f)
{
	int sz = 0;
	const char* data = (const char*)_data;
	
	do
	{
		int w = fwrite(data + sz*sizeelement, sizeelement, numelement - sz, f);
		sz += w;

		if(w == 0)
		{
			return 0;
		}
	}while(sz < numelement);

	return sz;
}

static int sure_fread(void* _data, int sizeelement, int numelement, FILE* f)
{
	char* data = (char*)_data;
	int sz = 0;
	
	do
	{
		int r = fread(data + sz*sizeelement, sizeelement, numelement - sz, f);
		sz += r;

		if(r == 0)
		{
			return 0;
		}
	}while(sz < numelement);
	
	return sz;
}


static int l_checkpoint_save(lua_State* L)
{
	const int n = lua_gettop(L) - 1;
	
	if(!lua_isstring(L, 1))
	{
		return luaL_error(L, "checkpointSave must have a filename as the first argument");
	}
	
	const char* fn = lua_tostring(L, 1);
	FILE* f = fopen(fn, "w");
	
	if(!f)
	{
		return luaL_error(L, "failed to open `%s' for writing", fn);
	}
	
	char header[128];
	for(int i=0; i<128; i++)
		header[i] = 0;
	
	snprintf(header, 128, "CHECKPOINT");
	sure_fwrite(header, 1, 128, f); //write header
	sure_fwrite(&n, sizeof(int), 1, f); //write number of variables
	
	for(int i=2; i<=n+1; i++)
	{
		int size;
		char* buf = exportLuaVariable(L, i, &size);
		int sz = 0;
		
		if(!sure_fwrite(&size, sizeof(int), 1, f))
		{
			fclose(f);
			return luaL_error(L, "failed in write\n");
		}
		if(!sure_fwrite(buf, 1, size, f))
		{
			fclose(f);
			return luaL_error(L, "failed in write\n");
		}
		
		free(buf);
	}
	
	fclose(f);
	return 0;
}

static int l_checkpoint_load(lua_State* L)
{
	if(!lua_isstring(L, 1))
	{
		return luaL_error(L, "checkpointLoad must have a filename as the first argument");
	}
	
	const char* fn = lua_tostring(L, 1);
	FILE* f = fopen(fn, "r");
	
	if(!f)
	{
		return luaL_error(L, "failed to open `%s' for reading", fn);
	}
	lua_pop(L, lua_gettop(L));
	
	
	char header[128];
	int n;

	sure_fread(header, 1,         128, f); //should be CHECKPOINT\0\0\0...
	sure_fread(    &n, sizeof(int), 1, f); //read number of variables

	for(int i=0; i<n; i++)
	{
		int size;
		if(!sure_fread(&size, sizeof(int), 1, f))
		{
			fclose(f);
			return luaL_error(L, "failed in read\n");
		}

		char* buf = (char*)malloc(size+1);
		if(!sure_fread(buf, 1, size, f))
		{
			fclose(f);
			return luaL_error(L, "failed in read\n");
		}
		
		importLuaVariable(L, buf, size);

		free(buf);
	}
	
	fclose(f);
	return n;
}


void registerCheckPoint(lua_State* L)
{
	lua_pushcfunction(L, l_checkpoint_save);
	lua_setglobal(L, "checkpointSave");
	
	lua_pushcfunction(L, l_checkpoint_load);
	lua_setglobal(L, "checkpointLoad");
}

extern "C"
{
CHECKPOINT_API int lib_register(lua_State* L);
CHECKPOINT_API int lib_version(lua_State* L);
CHECKPOINT_API const char* lib_name(lua_State* L);
CHECKPOINT_API int lib_main(lua_State* L);
}

CHECKPOINT_API int lib_register(lua_State* L)
{
	registerCheckPoint(L);
	return 0;
}

CHECKPOINT_API int lib_version(lua_State* L)
{
	return __revi;
}

CHECKPOINT_API const char* lib_name(lua_State* L)
{
	return "CheckPoint";
}

CHECKPOINT_API int lib_main(lua_State* L)
{
	return 0;
}

