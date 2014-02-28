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

#ifndef CHECKPOINTER_H
#define CHECKPOINTER_H

extern "C" {
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
}
#include <vector>
#include "luabaseobject.h"
#include <stdlib.h>

#include <string>
using namespace std;

class Checkpointer : public LuaBaseObject
{
public:
	Checkpointer();
	~Checkpointer();

	LINEAGE1("Checkpointer")
	static const luaL_Reg* luaMethods();
	virtual int luaInit(lua_State* L);
	static int help(lua_State* L);

	virtual void encode(buffer* b);
	virtual int  decode(buffer* b);

	int compare(lua_State* L, int idx);

	void calculateChecksum();

	int l_add(lua_State* L, int idx);
	int l_get(lua_State* L);
	int l_gettable(lua_State* L);
	int l_checksum(lua_State* L);
	int l_tostring(lua_State* L);
    int l_copy(lua_State* L);

	int l_fromstring(lua_State* L, int idx);


	int l_savetofile(lua_State* L);
	int l_loadfromfile(lua_State* L);
	int l_options(lua_State* L);

	int internalStateSize();


	void clear();

	buffer* b;
	int n;
	int end_pos;

	char* data();
	char* setData(char* d);

	unsigned long checksum;
	bool has_checksum;

	int  operate_data(lua_State* L, int idx); 
	int deoperate_data(lua_State* L);

	int currentState(const char* d = 0);
	static int currentStateS(const char* d = 0);


	std::string debug_file;
};

int checkpointer_register(lua_State* L);

// 
#endif

