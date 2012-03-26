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

#include "spinoperation.h"
#include "spinsystem.h"
#define CLAMP(x, m) ((x<0)?0:(x>m?m:x))

using namespace std;
int lua_getNint(lua_State* L, int N, int* vec, int pos, int def);

SpinOperation::SpinOperation(std::string Name, int Slot, int NX, int NY, int NZ, int etype)
	: LuaBaseObject(etype), nx(NX), ny(NY), nz(NZ), operationName(Name), slot(Slot)
{
	nxyz = nx * ny * nz;
}

SpinOperation::~SpinOperation()
{
	
}

const string& SpinOperation::name()
{
	return operationName;
}
	
void SpinOperation::markSlotUsed(SpinSystem* ss)
{
	ss->slot_used[slot] = true;
}

int SpinOperation::getSite(int x, int y, int z)
{
	x = (x+10*nx) % nx;
	y = (y+10*ny) % ny;
	z = (z+10*nz) % nz;

	return x + nx*y + nx*ny*z;
}

bool SpinOperation::member(int px, int py, int pz)
{
	if(px < 0 || py < 0 || pz < 0)
		return false;

	if(px >= nx || py >= ny || pz >= nz)
		return false;
	
	return true;
}

int  SpinOperation::getidx(int px, int py, int pz)
{
	px = CLAMP(px, nx);
	py = CLAMP(py, ny);
	pz = CLAMP(pz, nz);
	
	return px + nx * (py + ny * pz);
}

int SpinOperation::luaInit(lua_State* L)
{
	int n[3];
	
	if(luaT_is<SpinSystem>(L, 1))
	{
		SpinSystem* ss = luaT_to<SpinSystem>(L, 1);
		n[0] = ss->nx;
		n[1] = ss->ny;
		n[2] = ss->nz;
	}
	else
	{
		lua_getNint(L, 3, n, 1, 1);
	}
	
	nx = n[0];
	ny = n[1];
	nz = n[2];
	nxyz = nx * ny * nz;
	
	return 0;
}

bool SpinOperation::apply(SpinSystem* ss)
{
	return 0;
}


int lua_getNint(lua_State* L, int N, int* vec, int pos, int def)
{
	if(lua_istable(L, pos))
	{
		for(int i=0; i<N; i++)
		{
			lua_pushinteger(L, i+1);
			lua_gettable(L, pos);
			if(lua_isnil(L, -1))
			{
				vec[i] = def;
			}
			else
			{
				vec[i] = lua_tointeger(L, -1);
			}
			lua_pop(L, 1);
		}
		return 1;
	}
	
	for(int i=0; i<N; i++)
	{
		if(lua_isnumber(L, pos+i))
		{
			vec[i] = lua_tointeger(L, pos+i);
		}
		else
		{
			vec[i] = def;
		}
//			return -1;
	}
	
	return N;
}

int lua_getnewargs(lua_State* L, int* vec, int pos)
{
	if(lua_istable(L, pos))
	{
		for(int i=0; i<3; i++)
		{
			lua_pushinteger(L, i+1);
			lua_gettable(L, pos);
			if(lua_isnil(L, -1))
			{
				vec[i] = 1;
			}
			else
			{
				vec[i] = lua_tointeger(L, -1);
			}
			lua_pop(L, 1);
		}
		return 1;
	}
	
	if(luaT_is<SpinSystem>(L, pos))
	{
		SpinSystem* ss = luaT_to<SpinSystem>(L, pos);
		vec[0] = ss->nx;
		vec[1] = ss->ny;
		vec[2] = ss->nz;
		return 1;
	}

	vec[0] = 1;
	vec[1] = 1;
	vec[2] = 1;

	for(int i=0; i<3; i++)
	{
		if(lua_isnumber(L, pos+i))
		{
			vec[i] = lua_tointeger(L, pos+i);
		}
		else
			return 3;
	}
	
	return 3;
}


int lua_getNdouble(lua_State* L, int N, double* vec, int pos, double def)
{
	if(lua_istable(L, pos))
	{
		for(int i=0; i<N; i++)
		{
			lua_pushinteger(L, i+1);
			lua_gettable(L, pos);
			if(lua_isnil(L, -1))
			{
				vec[i] = def;
			}
			else
			{
				vec[i] = lua_tonumber(L, -1);
			}
			lua_pop(L, 1);
		}
		return 1;
	}
	
	for(int i=0; i<N; i++)
	{
		if(lua_isnumber(L, pos+i))
		{
			vec[i] = lua_tonumber(L, pos+i);
		}
		else
			return -1;
	}
	
	return N;
}

#include "spinsystem.h"
static int l_apply(lua_State* L)
{
	LUA_PREAMBLE(SpinOperation,so,1);
	LUA_PREAMBLE(SpinSystem,ss,2);
	
	if(!so->apply(ss))
		return luaL_error(L, so->errormsg.c_str());
	return 0;
}

static int l_member(lua_State* L)
{
	LUA_PREAMBLE(SpinOperation,so,1);

	int vec[3];
	lua_getNint(L, 3, vec, 2, 1);
	
	if(so->member(vec[0]-1, vec[1]-1, vec[2]-1))
		lua_pushboolean(L, 1);
	else
		lua_pushboolean(L, 0);

	return 1;
}

static int l_tostring(lua_State* L)
{
	LUA_PREAMBLE(SpinOperation,so,1);
	lua_pushfstring(L, "%s (%dx%dx%d)", so->lineage(0), so->nx, so->ny, so->nz);
	return 1;
}

int SpinOperation::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Abstruct base class for Spin Operations");
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	if(!lua_iscfunction(L, 1))
	{
		return luaL_error(L, "help expect zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
		
	if(func == l_member)
	{
		lua_pushstring(L, "Test if the given site index is part of the operator");
		lua_pushstring(L, "1 *3Vector* (Integers): Index of site to test");
		lua_pushstring(L, "1 Boolean: Result of test");
		return 3;
	}
	if(func == l_apply)
	{
		lua_pushstring(L, "Apply the operator to the SpinSystem");
		lua_pushstring(L, "1 SpinSystem: System that will receive the resulting fields");
		lua_pushstring(L, "");
		return 3;
	}

	return LuaBaseObject::help(L);
}

static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* SpinOperation::luaMethods()
{
	if(m[127].name)
		return m;

	static const luaL_Reg _m[] =
	{
		{"__tostring",   l_tostring},
		{"member",       l_member},
		{"apply",        l_apply},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}

