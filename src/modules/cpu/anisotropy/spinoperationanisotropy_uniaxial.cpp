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

#include "spinoperationanisotropy_uniaxial.h"
#include "spinsystem.h"

#include <stdlib.h>

AnisotropyUniaxial::AnisotropyUniaxial(int nx, int ny, int nz)
	: SpinOperation(nx, ny, nz, hash32(AnisotropyUniaxial::typeName()))
{
	setSlotName("UniaxialAnisotropy");
	ops = 0;
	//size = nx*ny*nz;
	size = 0;
	init();
}


int AnisotropyUniaxial::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	size = 0;
// 	size = nx*ny*nz;
	init();
	return 0;
}

void AnisotropyUniaxial::init()
{
	num = 0;
	if(size < 0)
		size = 1;

	ops = (ani*)malloc(sizeof(ani) * size);
}

void AnisotropyUniaxial::deinit()
{
	if(ops)
	{
		free(ops);
	}
	size = 0;
	ops = 0;
}



static bool myfunction(AnisotropyUniaxial::ani* i,AnisotropyUniaxial::ani* j)
{
	return (i->site<j->site);
}

#include <algorithm>    // std::sort
#include <vector>       // std::vector
using namespace std;
// this is messy but more efficient than before
int AnisotropyUniaxial::merge()
{
	fprintf(stderr, "AnisotropyUniaxial merge() is unimplemented\n");
#if 0
	if(num == 0)
		return 0;

	int original_number = num;
	
	vector<ani*> new_ops;
	
	for(int i=0; i<num; i++)
	{
		new_ops.push_back(&ops[i]);
	}
	sort (new_ops.begin(), new_ops.end(), myfunction);

	ani* new_ops2 = (ani*) malloc(sizeof(ani)*size);
	int new_num2 = num;
	
	for(unsigned int i=0; i<new_ops.size(); i++)
	{
		memcpy(&new_ops2[i], new_ops[i], sizeof(ani));
	}
	
	int current = 0;
	
	num = 1;
	
	// put in the 1st site
	memcpy(&ops[0], &new_ops2[0], sizeof(ani));
	
	for(int i=1; i<new_num2; i++)
	{
		if(new_ops2[i].site == ops[current].site)
		{
			ops[current].strength += new_ops2[i].strength;
		}
		else
		{
			current++;
			num++;
			memcpy(&ops[current], &new_ops2[i], sizeof(ani));
		}
	}
	
	

	int delta = original_number - num;
	free(new_ops2);
	return delta;
#endif
	return 0;
}



bool AnisotropyUniaxial::getAnisotropy(int site, double* axis, double& K1, double& K2)
{
	for(int i=0; i<num; i++)
	{
		if(ops[i].site == site)
		{
			memcpy(axis, ops[i].axis, sizeof(double)*3);
			K1 = ops[i].K[0];
			K2 = ops[i].K[1];
			return true;
		}
	}
	return false;
}

static void cross(double* a, double* b, double* c)
{
	c[0] = a[1]*b[2] - a[2]*b[1];
	c[1] = a[2]*b[0] - a[0]*b[2];
	c[2] = a[0]*b[1] - a[1]*b[0];
}
static double dot(double* a, double* b)
{
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}
static void scale(double s, double* a)
{
	a[0] *= s;
	a[1] *= s;
	a[2] *= s;
}

void AnisotropyUniaxial::addAnisotropy(int site, double* axis, double K1, double K2)
{
	if(num == size)
	{
		if(size == 0)
			size = 32;
		else
			size = size * 2;
		ops = (ani*)realloc(ops, sizeof(ani) * size);
	}
	
	ops[num].site = site;
	memcpy(ops[num].axis, axis, sizeof(double)*3);
	
	const double length = dot(ops[num].axis, ops[num].axis);
	if(length > 0)
		scale(1.0/length, ops[num].axis);
		
	ops[num].K[0] = K1;
	ops[num].K[1] = K2;
	num++;
}

void AnisotropyUniaxial::encode(buffer* b)
{
	SpinOperation::encode(b); //nx,ny,nz,global_scale

	char version = 0;
	encodeChar(version, b);
	
	encodeInteger(num, b);
	for(int i=0; i<num; i++)
	{
		encodeInteger(ops[i].site, b);
		for(int coordinate=0; coordinate<3; coordinate++)
			encodeDouble(ops[i].axis[coordinate], b);

		for(int j=0; j<2; j++)
			encodeDouble(ops[i].K[j], b);
	}
}

int AnisotropyUniaxial::decode(buffer* b)
{
	deinit();
	SpinOperation::decode(b); //nx,ny,nz,global_scale
	
	char version = decodeChar(b);
	
	if(version == 0)
	{
		num = decodeInteger(b);
		size = num;
		init();
		
		for(int i=0; i<size; i++)
		{
			const int site = decodeInteger(b);
			
			double a[3];
			double K[2];
			
			for(int coordinate=0; coordinate<3; coordinate++)
				a[coordinate] = decodeDouble(b);
			
			for(int j=0; j<2; j++)
				K[j] = decodeDouble(b);

			addAnisotropy(site, a, K[0], K[1]);
		}
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}
	return 0;
}


AnisotropyUniaxial::~AnisotropyUniaxial()
{
	deinit();
}


// E_anis = - K1 * <axis, m>^2 - K2 * <axis, m>^4
// K1 = Second-order phenomenological anisotropy constant
// K2 = Fourth-order phenomenological anisotropy constant

// H_ani = [ 2 K1 (axis . s) + 4 K2 (axis . s)^3 ] axis

bool AnisotropyUniaxial::apply(SpinSystem* ss)
{
	int slot = markSlotUsed(ss);

	dArray& hx = (*ss->hx[slot]);
	dArray& hy = (*ss->hy[slot]);
	dArray& hz = (*ss->hz[slot]);

	dArray& x = (*ss->x);
	dArray& y = (*ss->y);
	dArray& z = (*ss->z);

	hx.zero();
	hy.zero();
	hz.zero();

// #pragma omp for schedule(static)
	for(int j=0; j<num; j++)
	{
		const ani& op = ops[j];
		const int i = op.site;
		const double ms = (*ss->ms)[i];
		if(ms > 0)
		{
			// H_ani = [ 2 K1 (axis . s) + 4 K2 (axis . s)^3 ] axis

			const double sx = x[i] / ms;
			const double sy = y[i] / ms;
			const double sz = z[i] / ms;
			
			const double nx = op.axis[0];
			const double ny = op.axis[1];
			const double nz = op.axis[2];
			
			const double K1 = op.K[0];
			const double K2 = op.K[1];
			
			const double SpinDotAxis  = sx*nx + sy*ny + sz*nx;
			const double SpinDotAxis3 = SpinDotAxis * SpinDotAxis * SpinDotAxis;
			
			const double magnitude = 2.0 * K1 * SpinDotAxis + 4.0 * K2 * SpinDotAxis3;

			hx[i] += nx * magnitude * global_scale;
			hy[i] += ny * magnitude * global_scale;
			hz[i] += nz * magnitude * global_scale;
		}
	}
	return true;
}







static int l_get(lua_State* L)
{
	LUA_PREAMBLE(AnisotropyUniaxial, ani, 1);

	double n[3], K1, K2;

	int p[3];
	int r1 = lua_getNint(L, 3, p, 2, 1);

	if(r1<0)
		return luaL_error(L, "invalid site format");
	
	if(!ani->member(p[0]-1, p[1]-1, p[2]-1))
		return luaL_error(L, "site is not part of system");

	int idx = ani->getidx(p[0]-1, p[1]-1, p[2]-1);
	

	if(!ani->getAnisotropy(idx, n, K1, K2))
	{
		lua_newtable(L);
		lua_pushinteger(L, 1); lua_pushnumber(L, 0); lua_settable(L, -3);
		lua_pushinteger(L, 2); lua_pushnumber(L, 0); lua_settable(L, -3);
		lua_pushinteger(L, 3); lua_pushnumber(L, 1); lua_settable(L, -3);
		lua_pushnumber(L, 0);
		lua_pushnumber(L, 0);
	}
	else
	{
		lua_newtable(L);
		lua_pushinteger(L, 1); lua_pushnumber(L, n[0]); lua_settable(L, -3);
		lua_pushinteger(L, 2); lua_pushnumber(L, n[1]); lua_settable(L, -3);
		lua_pushinteger(L, 3); lua_pushnumber(L, n[2]); lua_settable(L, -3);
		lua_pushnumber(L, K1);
		lua_pushnumber(L, K2);
	}
	return 3;
}

static int l_numofax(lua_State* L)
{
	LUA_PREAMBLE(AnisotropyUniaxial, ani, 1);
	lua_pushinteger(L, ani->num);
	return 1;
}


static int l_axisat(lua_State* L)
{
	LUA_PREAMBLE(AnisotropyUniaxial, ani, 1);
	
	int idx = lua_tointeger(L, 2) - 1;

	if(idx < 0 || idx >= ani->num)
		return luaL_error(L, "Invalid axis index");
	

	const int site = ani->ops[idx].site;
	const double* axis = ani->ops[idx].axis;
	const double K1 = ani->ops[idx].K[0];
	const double K2 = ani->ops[idx].K[1];
	
	int x,y,z;
	ani->idx2xyz(site, x, y, z);

	lua_newtable(L);
	lua_pushinteger(L, 1); lua_pushinteger(L, x+1); lua_settable(L, -3);
	lua_pushinteger(L, 2); lua_pushinteger(L, y+1); lua_settable(L, -3);
	lua_pushinteger(L, 3); lua_pushinteger(L, z+1); lua_settable(L, -3);
	
	lua_newtable(L);
	lua_pushinteger(L, 1); lua_pushnumber(L, axis[0]); lua_settable(L, -3);
	lua_pushinteger(L, 2); lua_pushnumber(L, axis[1]); lua_settable(L, -3);
	lua_pushinteger(L, 3); lua_pushnumber(L, axis[2]); lua_settable(L, -3);
	
	lua_pushnumber(L, K1);
	lua_pushnumber(L, K2);
	
	return 3;
}

static int l_add(lua_State* L)
{
	LUA_PREAMBLE(AnisotropyUniaxial, ani, 1);

	int p[3];

	int r1 = lua_getNint(L, 3, p, 2, 1);
	
	if(r1<0)
		return luaL_error(L, "invalid site format");
	
	if(!ani->member(p[0]-1, p[1]-1, p[2]-1))
		return luaL_error(L, "site (%d, %d, %d) is not part of operator (%dx%dx%d)", p[0], p[1], p[2], ani->nx, ani->ny, ani->nz);

	int idx = ani->getidx(p[0]-1, p[1]-1, p[2]-1);

	double a[3];	
	int r2 = lua_getNdouble(L, 3, a, 2+r1, 0);
	if(r2<0)
		return luaL_error(L, "invalid anisotropy direction");

	/* anisotropy axis is a unit vector */
	const double lena = sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
	
	if(lena > 0)
	{
		a[0] /= lena;
		a[1] /= lena;
		a[2] /= lena;
	}
	else
		return 0; //don't add ani
	
	double K1 = 0;
	double K2 = 0;

	if(lua_isnumber(L, 2+r1+r2))
		K1 = lua_tonumber(L, 2+r1+r2);
	else
		return luaL_error(L, "anisotropy needs strength");

	if(lua_isnumber(L, 2+r1+r2+1))
		K2 = lua_tonumber(L, 2+r1+r2);

	
	ani->addAnisotropy(idx, a, K1, K2);
	return 0;
}

static int l_mergeAxes(lua_State* L)
{
	LUA_PREAMBLE(AnisotropyUniaxial, ani, 1);
	lua_pushinteger(L, ani->merge());
	return 1;	
}

int AnisotropyUniaxial::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Computes the uniaxial anisotropy fields for a *SpinSystem* as the derivative of the following energy expression.\n"
					"<pre>E_anis = - K1 * &lt;axis, m&gt;^2 - K2 * &lt;axis, m&gt;^4\n"
					"K1 = Second-order phenomenological anisotropy constant\n"
					"K2 = Fourth-order phenomenological anisotropy constant</pre>"
		);
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
		
	if(func == l_add)
	{
		lua_pushstring(L, "Add a lattice site to the anisotropy calculation");
		lua_pushstring(L, "2 *3Vector*s, 1 or 2 numbers: The first *3Vector* defines a lattice site, the second defines an axis and is normalized. The first number is required and defines the strength of the second order phenomenological constant. The second number is optional with a default of 0 and defines the fourth order phenomenological anisotropy constant.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_get)
	{
		lua_pushstring(L, "Fetch the anisotropy direction and magnitudes at a given site.");
		lua_pushstring(L, "1 *3Vector*: The *3Vector* defines a lattice site.");
		lua_pushstring(L, "1 Table, 2 Numbers: The table defines the normal axis, the two numbers are the K1 and K2 for the site.");
		return 3;
	}
	
	if(func == l_axisat)
	{
		lua_pushstring(L, "Return the site, axis and strengths (K1, K2) at the given index.");
		lua_pushstring(L, "1 Integer: Index of the axis.");
		lua_pushstring(L, "1 Table of 3 Integers, 1 Table of 3 Numbers, 2 Numbers: Coordinates of the site, direction of the axis and strengths (K1, K2) of the axis.");
		return 3;	
	}
	
	if(func == l_numofax)
	{
		lua_pushstring(L, "Return the number of axes in the operator");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Number of axes.");
		return 3;		
	}
	
	if(func == l_mergeAxes)
	{
		lua_pushstring(L, "Combine common site-axes into a single axis with a combined strength");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;			
	}
	
	
	return SpinOperation::help(L);
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* AnisotropyUniaxial::luaMethods()
{
	if(m[127].name)return m;

	merge_luaL_Reg(m, SpinOperation::luaMethods());
	static const luaL_Reg _m[] =
	{
		{"add",          l_add},
		{"get",          l_get},
		{"numberOfAxes", l_numofax},
		{"axis", l_axisat},
		{"mergeAxes", l_mergeAxes},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}


