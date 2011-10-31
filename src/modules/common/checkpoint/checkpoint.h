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

#ifdef WIN32
 #define strcasecmp(A,B) _stricmp(A,B)
 #define strncasecmp(A,B,C) _strnicmp(A,B,C)
 #pragma warning(disable: 4251)
 #define snprintf _snprintf

 #ifdef CHECKPOINT_EXPORTS
  #define CHECKPOINT_API __declspec(dllexport)
 #else
  #define CHECKPOINT_API __declspec(dllimport)
 #endif
#else
 #define CHECKPOINT_API 
#endif


extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>

CHECKPOINT_API int lib_register(lua_State* L);
CHECKPOINT_API int lib_version(lua_State* L);
CHECKPOINT_API const char* lib_name(lua_State* L);
CHECKPOINT_API void lib_main(lua_State* L, int argc, char** argv);
}
