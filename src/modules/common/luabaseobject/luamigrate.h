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
extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

LUABASEOBJECT_API char* exportLuaVariable(lua_State* L, int index,   int* chunksize);
LUABASEOBJECT_API int   importLuaVariable(lua_State* L, char* chunk, int  chunksize);

LUABASEOBJECT_API void _exportLuaVariable(lua_State* L, int index, buffer* b);
LUABASEOBJECT_API int _importLuaVariable(lua_State* L, buffer* b);
}
