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

#include "luamigrate.h"
#include "checkpoint.h"
#include <stdio.h>
#include <stdlib.h>

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
	for(int i=0; i<128; i++) header[i] = 0;
	
	snprintf(header, 128, "CHECKPOINT");
	fwrite(header, 1, 128, f); //write header
	
	fwrite(&n, 1, sizeof(int), f); //write number of variables
	
	for(int i=2; i<=n+1; i++)
	{
		int size;
		char* buf = exportLuaVariable(L, i, &size);

		printf("size %i = %i\n", i, size);
		fwrite(&size, 1, sizeof(int), f);
		fwrite(buf, 1, size, f);
		
		free(buf);
	}
	
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
	
	char header[128];
	fread(header, 1, 128, f); //should be CHECKPOINT\0\0\0...
	int n;
	fread(&n, 1, sizeof(int), f); //read number of variables

	for(int i=0; i<n; i++)
	{
		int size;
		fread(&size, 1, sizeof(int), f);
			
		char* buf = (char*)malloc(size+1);
		fread(buf, 1, size, f);
		
		importLuaVariable(L, buf, size);

		free(buf);
	}
	return n;
}


void registerCheckPoint(lua_State* L)
{
	lua_pushcfunction(L, l_checkpoint_save);
	lua_setglobal(L, "checkpointSave");
	
	lua_pushcfunction(L, l_checkpoint_load);
	lua_setglobal(L, "checkpointLoad");
}

int lib_register(lua_State* L)
{
	registerCheckPoint(L);
	return 0;
}
