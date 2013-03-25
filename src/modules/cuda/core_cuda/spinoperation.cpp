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
	global_scale = 1.0;
	registerWS();
}

SpinOperation::~SpinOperation()
{
	unregisterWS();
}

int SpinOperation::luaInit(lua_State* L)
{
	LuaBaseObject::luaInit(L);
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


void SpinOperation::encode(buffer* b)
{
	char version = 0;
	encodeChar(version, b);
	encodeInteger(nx, b);
	encodeInteger(ny, b);
	encodeInteger(nz, b);
	encodeDouble(global_scale, b);
}

int SpinOperation::decode(buffer* b)
{
	char version = decodeChar(b);
	if(version == 0)
	{
		nx = decodeInteger(b);
		ny = decodeInteger(b);
		nz = decodeInteger(b);
		nxyz = nx*ny*nz;
		global_scale = decodeDouble(b);
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}

	return 0;
}

double*  SpinOperation::getVectorOfValues(SpinSystem** sss, int n, const char* tag, const char _data, const double scale)
{
	double *d_v, *h_v;
	const char data = _data | 0x20; // make data lower case

    getWSMemD(&d_v, sizeof(double)*n, hash32(tag));
    getWSMemH(&h_v, sizeof(double)*n, hash32(tag));
	
	switch(data)
	{
	case 'a':
		for(int i=0; i<n; i++)
			h_v[i] = sss[i]->alpha;
		break;
	case 'g':
		for(int i=0; i<n; i++)
			h_v[i] = sss[i]->gamma;
		break;
	case 'd':
		for(int i=0; i<n; i++)
			h_v[i] = sss[i]->dt;
		break;
	default:
		fprintf(stderr, "(%s:%i) don't know what to do with %c\n", __FILE__, __LINE__, _data);
	}
	
	for(int i=0; i<n; i++)
	{
		h_v[i] *= scale;
	}
	
	memcpy_h2d(d_v, h_v, sizeof(double)*n);
	return d_v;
}

double** SpinOperation::getVectorOfVectors(SpinSystem** sss, int n, const char* tag, const char _data, const char _component, const int field)
{
	double **d_v, **h_v;

	char data = _data | 0x20; // make data lower case
	char component = _component | 0x20; // make component lower case

    getWSMemD(&d_v, sizeof(double*)*n, hash32(tag));
    getWSMemH(&h_v, sizeof(double*)*n, hash32(tag));

	switch(data)
	{
	case 'h':
		for(int i=0; i<n; i++)
		{
			if(component == 'x') h_v[i] = sss[i]->hx[field]->ddata();
			if(component == 'y') h_v[i] = sss[i]->hy[field]->ddata();
			if(component == 'z') h_v[i] = sss[i]->hz[field]->ddata();
		}
		break;
	case 's':
		for(int i=0; i<n; i++)
		{
			if(component == 'x') h_v[i] = sss[i]->x->ddata();
			if(component == 'y') h_v[i] = sss[i]->y->ddata();
			if(component == 'z') h_v[i] = sss[i]->z->ddata();
			if(component == 'm') h_v[i] = sss[i]->ms->ddata();
		}
		break;
	case 'a':
		for(int i=0; i<n; i++)
		{
			if(sss[i]->site_alpha)
				h_v[i] = sss[i]->site_alpha->ddata();
			else
				h_v[i] = 0;
		}
		break;
	case 'g':
		for(int i=0; i<n; i++)
		{
			if(sss[i]->site_gamma)
				h_v[i] = sss[i]->site_gamma->ddata();
			else
				h_v[i] = 0;
		}
		break;
	default:
		fprintf(stderr, "(%s:%i) don't know what to do with %c\n", __FILE__, __LINE__, _data);
	}
	
	memcpy_h2d(d_v, h_v, sizeof(double*)*n);

	return d_v;
}

void SpinOperation::getSpinSystemsAtPosition(lua_State* L, int pos, vector<SpinSystem*>& sss)
{
	int initial_size = lua_gettop(L);
	
	if(pos < 0)
	{
		pos = initial_size + pos + 1;
	}
	if(lua_istable(L, pos))
	{
		if(lua_istable(L, pos))
		{
			lua_pushnil(L);
			while(lua_next(L, pos))
			{
				SpinSystem* ss = luaT_to<SpinSystem>(L, -1);
				if(ss)
					sss.push_back(ss);
				lua_pop(L, 1);
			}
		}
	}
	else
	{
		if(luaT_is<SpinSystem>(L, pos))
		{
			sss.push_back(luaT_to<SpinSystem>(L, pos));
		}
	}

	
	while(lua_gettop(L) > initial_size)
		lua_pop(L, 1);
}

const string& SpinOperation::name()
{
	return operationName;
}
	
void SpinOperation::markSlotUsed(SpinSystem* ss)
{
	ss->ensureSlotExists(slot);
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

void SpinOperation::idx2xyz(int idx, int& x, int& y, int& z) const 
{
	while(idx < 0)
		idx += 10*nxyz;
	idx %= nxyz;
	
	z = idx / (nx*ny);
	idx -= z*nx*ny;
	y = idx / nx;
	x = idx - y*nx;
}

bool SpinOperation::apply(SpinSystem** sss, int n)
{
	errormsg="No multi-spinsystem apply method is defined for this operator";
	return false;
}


bool SpinOperation::apply(SpinSystem* ss)
{
	errormsg="No apply method is defined for this operator";
	return false;
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
	
	vector<SpinSystem*> sss;
	so->getSpinSystemsAtPosition(L, 2, sss);
	
	if(sss.size() == 0)
	{
		return 0; //don't need to do anything
	}
	if(sss.size() == 1)
	{
		if(!so->apply(sss[0]))
			return luaL_error(L, so->errormsg.c_str());
	}
	if(sss.size() >  1)
	{
		if(!so->apply(&(sss[0]), sss.size()))
			return luaL_error(L, so->errormsg.c_str());
	}
	
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

static int l_setscale(lua_State* L)
{
	LUA_PREAMBLE(SpinOperation,so,1);
	so->global_scale = lua_tonumber(L, 2);
	return 0;
}
static int l_getscale(lua_State* L)
{
	LUA_PREAMBLE(SpinOperation,so,1);
	lua_pushnumber(L, so->global_scale);
	return 1;
}

static int l_nx(lua_State* L)
{
	LUA_PREAMBLE(SpinOperation,so,1);
	lua_pushinteger(L, so->nx);
	return 1;
}
static int l_ny(lua_State* L)
{
	LUA_PREAMBLE(SpinOperation,so,1);
	lua_pushinteger(L, so->ny);
	return 1;
}
static int l_nz(lua_State* L)
{
	LUA_PREAMBLE(SpinOperation,so,1);
	lua_pushinteger(L, so->nz);
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
	
	if(!lua_isfunction(L, 1))
	{
		return luaL_error(L, "(%s:%i) Help expects zero arguments or 1 function.", __FILE__, __LINE__);
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
		lua_pushstring(L, "Apply the operator to the SpinSystem or Table of SpinSystems");
		lua_pushstring(L, "1 *SpinSystem* or 1 Table of SpinSystems: System that will receive the resulting fields. If a table of SpinSystems is provided then this operator will be applied to each SpinSystem.");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_setscale)
	{
		lua_pushstring(L, "Set a scale to field calculatons (default value is 1.0)");
		lua_pushstring(L, "1 Number: The value of the new scale");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == l_getscale)
	{
		lua_pushstring(L, "Get the scale applied to field calculatons (default value is 1.0)");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: The scale");
		return 3;
	}
	
	if(func == l_nx)
	{
		lua_pushstring(L, "Get the size in the x direction that this operator was created with.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: size");
		return 3;
	}
	if(func == l_ny)
	{
		lua_pushstring(L, "Get the size in the y direction that this operator was created with.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: size");
		return 3;
	}
	if(func == l_nz)
	{
		lua_pushstring(L, "Get the size in the z direction that this operator was created with.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: size");
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
		{"setScale",     l_setscale},
		{"scale",        l_getscale},
		{"nx",        l_nx},
		{"ny",        l_ny},
		{"nz",        l_nz},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}

