/******************************************************************************
* Copyright (C) 2008-2014 Jason Mercer.  All rights reserved.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
******************************************************************************/


#ifdef _CREATE_LIBRARY

#include "luamigrate.h"
#include "checkpointer.h"
#include "info.h"

#ifndef CHECKPOINT_API
#define CHECKPOINT_API
#endif

extern "C"
{
CHECKPOINT_API int lib_register(lua_State* L);
CHECKPOINT_API int lib_version(lua_State* L);
CHECKPOINT_API const char* lib_name(lua_State* L);
CHECKPOINT_API int lib_main(lua_State* L);
}

CHECKPOINT_API int lib_register(lua_State* L)
{
	checkpointer_register(L);

	return 0;
}

CHECKPOINT_API int lib_version(lua_State* L)
{
	return __revi;
}

CHECKPOINT_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "CheckPoint";
#else
	return "CheckPoint-Debug";
#endif
}

#include "checkpoint_main.h"


CHECKPOINT_API int lib_main(lua_State* L)
{
    luaL_dofile_checkpoint_main(L);

    return 0;
}

#endif
