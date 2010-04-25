#include "spinoperation.h"
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
			return -1;
	}
	
	return N;
}

int lua_getnewargs(lua_State* L, int* vec3, int pos)
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
				vec[i] = lua_tonumber(L, -1);
			}
			lua_pop(L, 1);
		}
		return 1;
	}

	vec[0] = 1;
	vec[1] = 1;
	vec[2] = 1;

	for(int i=0; i<3; i++)
	{
		if(lua_isnumber(L, pos+i))
		{
			vec[i] = lua_tonumber(L, pos+i);
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

