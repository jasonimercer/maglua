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
#else
#include <windows.h>
inline char* exportLuaVariable(lua_State* L, int index,   int* chunksize)
{
	typedef char* (*func)(lua_State*, int, int*); 
	static func exfunc = 0;
	if(!exfunc)
	{
		exfunc = (func) GetProcAddress(GetModuleHandle(NULL), "exportLuaVariable");
	}

	if(!exfunc)
	{
		printf("failed to load exportLuaVariable\n");
		return NULL;
	}
	return exfunc(L, index, chunksize);
}
inline int  importLuaVariable(lua_State* L, char* chunk, int  chunksize)
{
	typedef int(*func)(lua_State*, char*, int); 
	static func imfunc = 0;
	if(!imfunc)
	{
		imfunc = (func) GetProcAddress(GetModuleHandle(NULL), "importLuaVariable");
	}

	if(!imfunc)
	{
		printf("failed to load importLuaVariable\n");
		return NULL;
	}
	return imfunc(L, chunk, chunksize);
}

#endif
#else
ENCODE_API char* exportLuaVariable(lua_State* L, int index,   int* chunksize);
ENCODE_API int   importLuaVariable(lua_State* L, char* chunk, int  chunksize);
#endif
