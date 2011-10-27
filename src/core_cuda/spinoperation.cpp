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

SpinOperation::SpinOperation(std::string Name, int Slot, int NX, int NY, int NZ, int etype)
	: Encodable(etype), nx(NX), ny(NY), nz(NZ), refcount(0), operationName(Name), slot(Slot)
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
	
	if(lua_isSpinSystem(L, pos))
	{
		SpinSystem* ss = lua_toSpinSystem(L, pos);
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

