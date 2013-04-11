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

#include "spinoperationanisotropy_cubic.h"
#include "spinsystem.h"

#include <stdlib.h>

AnisotropyCubic::AnisotropyCubic(int nx, int ny, int nz)
	: SpinOperation(nx, ny, nz, hash32(AnisotropyCubic::typeName()))
{
	ops = 0;
	//size = nx*ny*nz;
	size = 0;
	init();
}

const char* AnisotropyCubic::getSlotName()
{
	return "CubicAnisotropy";
}


int AnisotropyCubic::luaInit(lua_State* L)
{
	deinit();
	SpinOperation::luaInit(L); //gets nx, ny, nz, nxyz
	size = 0;
// 	size = nx*ny*nz;
	init();
	return 0;
}

void AnisotropyCubic::init()
{
	num = 0;
	if(size < 0)
		size = 1;

	ops = (ani*)malloc(sizeof(ani) * size);
}

void AnisotropyCubic::deinit()
{
	if(ops)
	{
		free(ops);
	}
	size = 0;
	ops = 0;
}



static bool myfunction(AnisotropyCubic::ani* i,AnisotropyCubic::ani* j)
{
	return (i->site<j->site);
}

#include <algorithm>    // std::sort
#include <vector>       // std::vector
using namespace std;
// this is messy but more efficient than before
int AnisotropyCubic::merge()
{
	fprintf(stderr, "AnisotropyCubic merge() is unimplemented\n");
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

bool AnisotropyCubic::getAnisotropy(int site, double* a1, double* a2, double* a3, double* K3)
{
	for(int i=0; i<num; i++)
	{
		if(ops[i].site == site)
		{
			memcpy(a1, ops[i].axis[0], sizeof(double)*3);
			memcpy(a2, ops[i].axis[1], sizeof(double)*3);
			memcpy(a3, ops[i].axis[2], sizeof(double)*3);
			memcpy(K3, ops[i].K,       sizeof(double)*3);
			return true;
		}
	}
	
	a1[0] = 1; a1[1] = 0; a1[2] = 0;
	a2[0] = 0; a2[1] = 1; a2[2] = 0;
	a3[0] = 0; a3[1] = 0; a3[2] = 1;
	
	K3[0] = 0;
	K3[1] = 0;
	K3[2] = 0;
	
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
// vector scale add
static void vsadd(double* s1, double a, double* s2, double* dest)
{
	for(int i=0; i<3; i++)
		dest[i] = s1[i] + a * s2[i];
}
static void scale(double s, double* a, double* dest=0)
{
	if(dest)
	{
		dest[0] = a[0] * s;
		dest[1] = a[1] * s;
		dest[2] = a[2] * s;
	}
	else
	{
		a[0] *= s;
		a[1] *= s;
		a[2] *= s;
	}
}

int AnisotropyCubic::addAnisotropy(int site, double* a1, double* a2, double* K3)
{
	if(num == size)
	{
		if(size == 0)
			size = 32;
		else
			size = size * 2;
		ops = (ani*)realloc(ops, sizeof(ani) * size);
	}
	
	//normalize vectors
	if(dot(a1,a1) == 0 || dot(a2,a2) == 0)
	{
		return 1; // 0 length vector = bad
	}
	
	scale(1.0/sqrt(dot(a1,a1)), a1);
	scale(1.0/sqrt(dot(a2,a2)), a2);
	
	if(fabs(dot(a1,a2)) == 1)
	{
		return 2; // colinear vectors = bad
	}
	
	double a3[3];
	cross(a1, a2, a3);
	scale(1.0/sqrt(dot(a3,a3)), a3);

	// first need to make sure a1, a2 are ortho (or ortho-able)
	if(dot(a1,a2) != 0)
	{
		// project vector a2 onto a3
		double a2_proj_a1[3];
		scale(dot(a1,a2), a1, a2_proj_a1);
		
		// subtract from 
		vsadd(a2, -1, a2_proj_a1, a2);
	}
	
	ops[num].site = site;
	memcpy(ops[num].axis[0], a1, sizeof(double)*3);
	memcpy(ops[num].axis[1], a2, sizeof(double)*3);
	
	cross(ops[num].axis[0], ops[num].axis[1], ops[num].axis[2]);

	for(int i=0; i<3; i++)
	{
		const double length = dot(ops[num].axis[i], ops[num].axis[i]);
		if(length > 0)
			scale(1.0/length, ops[num].axis[i]);
	}
	
	memcpy(ops[num].K, K3, sizeof(double)*3);
	num++;
}

void AnisotropyCubic::encode(buffer* b)
{
	SpinOperation::encode(b); //nx,ny,nz,global_scale

	char version = 0;
	encodeChar(version, b);
	
	encodeInteger(num, b);
	for(int i=0; i<num; i++)
	{
		encodeInteger(ops[i].site, b);
		for(int axis=0; axis<2; axis++) // JUST ENCODING 2 AXIS, 3RD IS COMPUTED
			for(int coordinate=0; coordinate<3; coordinate++)
				encodeDouble(ops[i].axis[axis][coordinate], b);

		for(int j=0; j<3; j++)
			encodeDouble(ops[i].K[j], b);
	}
}

int AnisotropyCubic::decode(buffer* b)
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
			
			double a[2][3];
			double K[3];
			
			for(int axis=0; axis<2; axis++) // JUST DECODING 2 AXIS, 3RD IS COMPUTED
				for(int coordinate=0; coordinate<3; coordinate++)
					a[axis][coordinate] = decodeDouble(b);
			
			for(int j=0; j<3; j++)
				K[j] = decodeDouble(b);

			addAnisotropy(site, a[0], a[1], K);
		}
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}
	return 0;
}


AnisotropyCubic::~AnisotropyCubic()
{
	deinit();
}


// E_anis = K1 * (<axis1,m>^2 <axis2,m>^2 + <axis1,m>^2 <axis3,m>^2 + <axis2,m>^2 <axis3,m>^2)
//        + K2 * (<axis1,m>^2 <axis2,m>^2 <axis3,m>^2)
//        + K3 * (<axis1,m>^4 <axis2,m>^4 + <axis1,m>^4 <axis3,m>^4 + <axis2,m>^4 <axis3,m>^4)

bool AnisotropyCubic::apply(SpinSystem* ss)
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

//  #pragma omp for schedule(static)
	for(int j=0; j<num; j++)
	{
		const ani& op = ops[j];
		const int i = op.site;
		const double ms = (*ss->ms)[i];
		if(ms > 0)
		{
			const double sx = x[i] / ms;
			const double sy = y[i] / ms;
			const double sz = z[i] / ms;
		
			const double n1x = op.axis[0][0];
			const double n1y = op.axis[0][1];
			const double n1z = op.axis[0][2];
			
			const double n2x = op.axis[1][0];
			const double n2y = op.axis[1][1];
			const double n2z = op.axis[1][2];
			
			const double n3x = op.axis[2][0];
			const double n3y = op.axis[2][1];
			const double n3z = op.axis[2][2];
			
			const double K1 = op.K[0];
			const double K2 = op.K[1];
			const double K3 = op.K[2];
			
			const double n1s = n1x*sx + n1y*sy + n1z*sz;
			const double n2s = n2x*sx + n2y*sy + n2z*sz;
			const double n3s = n3x*sx + n3y*sy + n3z*sz;
			
			const double n1s2 = n1s * n1s;
			const double n2s2 = n2s * n2s;
			const double n3s2 = n3s * n3s;

			const double n1s3 = n1s2 * n1s;
			const double n2s3 = n2s2 * n2s;
			const double n3s3 = n3s2 * n3s;

			const double n1s4 = n1s2 * n1s2;
			const double n2s4 = n2s2 * n2s2;
			const double n3s4 = n3s2 * n3s2;

// 			Hx = 2K1 (n1x n1s (n2s^2 + n3s^3) + n2x n2s (n1s^2 + n3s^2) + n3x n3s (n1s^2 + n2s^2)) +       //Term1
// 			     2K2 (n1s n2s n3s (n1x n2s n3s + n2x n1s n3s + n3x n1s n2s)) +                             //Term2
// 			     4K3 (n1x n1s^3 (n2s^4 + n3s^4) + n2x n2s^3 (n1s^4 + n3s^4) + n3x n3s^3 (n1s^4 + n2s^4))   //Term3
			
			const double t1x = 2.0 * K1 * (n1x * n1s * (n2s2 + n3s3) + n2x * n2s * (n1s2 + n3s2) + n3x * n3s * (n1s2 + n2s2));
			const double t1y = 2.0 * K1 * (n1y * n1s * (n2s2 + n3s3) + n2y * n2s * (n1s2 + n3s2) + n3y * n3s * (n1s2 + n2s2));
			const double t1z = 2.0 * K1 * (n1z * n1s * (n2s2 + n3s3) + n2z * n2s * (n1s2 + n3s2) + n3z * n3s * (n1s2 + n2s2));
			
			const double t2x = 2.0 * K2 * (n1s * n2s * n3s * (n1x * n2s * n3s + n2x * n1s * n3s + n3x * n1s * n2s));
			const double t2y = 2.0 * K2 * (n1s * n2s * n3s * (n1y * n2s * n3s + n2y * n1s * n3s + n3y * n1s * n2s));
			const double t2z = 2.0 * K2 * (n1s * n2s * n3s * (n1z * n2s * n3s + n2z * n1s * n3s + n3z * n1s * n2s));
			
			const double t3x = 4.0 * K3 * (n1x * n1s3 * (n2s4 + n3s4) + n2x * n2s3 * (n1s4 + n3s4) + n3x * n3s3 * (n1s4 + n2s4));
			const double t3y = 4.0 * K3 * (n1y * n1s3 * (n2s4 + n3s4) + n2y * n2s3 * (n1s4 + n3s4) + n3y * n3s3 * (n1s4 + n2s4));
			const double t3z = 4.0 * K3 * (n1z * n1s3 * (n2s4 + n3s4) + n2z * n2s3 * (n1s4 + n3s4) + n3z * n3s3 * (n1s4 + n2s4));
			
			hx[i] += global_scale * (t1x + t2x + t3x);
			hy[i] += global_scale * (t1y + t2y + t3y);
			hz[i] += global_scale * (t1z + t2z + t3z);
		}
	}
	return true;
}






static int l_get(lua_State* L)
{
	LUA_PREAMBLE(AnisotropyCubic, ani, 1);

	double a[3][3];
	double K[3];

	int p[3];
	int r1 = lua_getNint(L, 3, p, 2, 1);

	if(r1<0)
		return luaL_error(L, "invalid site format");
	
	if(!ani->member(p[0]-1, p[1]-1, p[2]-1))
		return luaL_error(L, "site is not part of system");

	int idx = ani->getidx(p[0]-1, p[1]-1, p[2]-1);
	

	ani->getAnisotropy(idx, a[0],a[1],a[2],K);
	
	for(int i=0; i<3; i++)
	{
		lua_newtable(L);
		lua_pushinteger(L, 1); lua_pushnumber(L, a[i][0]); lua_settable(L, -3);
		lua_pushinteger(L, 2); lua_pushnumber(L, a[i][1]); lua_settable(L, -3);
		lua_pushinteger(L, 3); lua_pushnumber(L, a[i][2]); lua_settable(L, -3);
	}
	lua_pushnumber(L, K[0]);
	lua_pushnumber(L, K[1]);
	lua_pushnumber(L, K[2]);

	return 6;
}



static int l_numofax(lua_State* L)
{
	LUA_PREAMBLE(AnisotropyCubic, ani, 1);
	lua_pushinteger(L, ani->num);
	return 1;
}


static int l_axisat(lua_State* L)
{
	LUA_PREAMBLE(AnisotropyCubic, ani, 1);
	
	int idx = lua_tointeger(L, 2) - 1;

	if(idx < 0 || idx >= ani->num)
		return luaL_error(L, "Invalid axis index");
	

	const int site = ani->ops[idx].site;
	double* K = ani->ops[idx].K;
	
	int x,y,z;
	ani->idx2xyz(site, x, y, z);

	lua_newtable(L);
	lua_pushinteger(L, 1); lua_pushinteger(L, x+1); lua_settable(L, -3);
	lua_pushinteger(L, 2); lua_pushinteger(L, y+1); lua_settable(L, -3);
	lua_pushinteger(L, 3); lua_pushinteger(L, z+1); lua_settable(L, -3);
	
	for(int i=0; i<3; i++)
	{
		lua_newtable(L);
		for(int j=0; j<3; j++)
		{
			lua_pushinteger(L, j+1);
			lua_pushnumber(L, ani->ops[idx].axis[i][j]);
			lua_settable(L, -3);
		}
	}
	
	for(int i=0; i<3; i++)
		lua_pushnumber(L, K[i]);
	
	return 7;
}

static int l_add(lua_State* L)
{
	LUA_PREAMBLE(AnisotropyCubic, ani, 1);

	int p[3];

	int r1 = lua_getNint(L, 3, p, 2, 1);
	
	if(r1<0)
		return luaL_error(L, "invalid site format");
	
	if(!ani->member(p[0]-1, p[1]-1, p[2]-1))
		return luaL_error(L, "site (%d, %d, %d) is not part of operator (%dx%dx%d)", p[0], p[1], p[2], ani->nx, ani->ny, ani->nz);

	int idx = ani->getidx(p[0]-1, p[1]-1, p[2]-1);

	double a[2][3];	
	int r2 = lua_getNdouble(L, 3, a[0], 2+r1, 0);
	if(r2<0)
		return luaL_error(L, "invalid anisotropy axis");
	int r3 = lua_getNdouble(L, 3, a[1], 2+r1+r2, 0);
	if(r3<0)
		return luaL_error(L, "invalid anisotropy axis");


	int t = 0;
	double K[3] = {0,0,0};
	for(int i=2+r1+r2+r3; i<=lua_gettop(L) && t < 3; i++)
	{
		if(lua_isnumber(L, i))
		{
			K[t] = lua_tonumber(L, i);
			t++;
		}
	}

	if(t == 0)
		return luaL_error(L, "anisotropy needs strength");
	
	if(ani->addAnisotropy(idx, a[0], a[1], K))
		return luaL_error(L, "Failed to add anisotropy, are your vectors colinear or empty?");
	return 0;
}

static int l_mergeAxes(lua_State* L)
{
	LUA_PREAMBLE(AnisotropyCubic, ani, 1);
	lua_pushinteger(L, ani->merge());
	return 1;	
}

int AnisotropyCubic::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Computes the cubic anisotropy fields for a *SpinSystem* as the derivative of the following energy expression.\n"
					"<pre>"
					"E_anis = K1 * (&lt;axis1,m&gt;^2 &lt;axis2,m&gt;^2 + &lt;axis1,m&gt;^2 &lt;axis3,m&gt;^2 + &lt;axis2,m&gt;^2 &lt;axis3,m&gt;^2)\n"
					"       + K2 * (&lt;axis1,m&gt;^2 &lt;axis2,m&gt;^2 &lt;axis3,m&gt;^2)\n"
					"       + K3 * (&lt;axis1,m&gt;^4 &lt;axis2,m&gt;^4 + &lt;axis1,m&gt;^4 &lt;axis3,m&gt;^4 + &lt;axis2,m&gt;^4 &lt;axis3,m&gt;^4)\n"
					"</pre>"
					"K1 = Fourth-order phenomenological anisotropy constant\n"
					"K2 = Sixth-order phenomenological anisotropy constant\n"
					"K3 = Eigth-order phenomenological anisotropy constant</pre>"
		);
		
		lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size"); 
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
		
	if(func == l_add)
	{
		lua_pushstring(L, "Add a lattice site to the anisotropy calculation");
		lua_pushstring(L, "3 *3Vector*s, 1 to 3 numbers: The first *3Vector* defines a lattice site, the second and third define an anisotropy axes which will be made unit vectors and crossed to create the 3rd cubic axis.  The numbers define the 4th, 6th and 8th order phenomenological constants, the first is required and the last two have default values of zero.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_get)
	{
		lua_pushstring(L, "Fetch the anisotropy direction and magnitude at a given site.");
		lua_pushstring(L, "1 *3Vector*: The *3Vector* defines a lattice site.");
		lua_pushstring(L, "3 Tables of 3 Nubmers, 3 Numbers: The tables define the axes and the numbers are the 4th, 6th and 8th order phenomenological constants.");
		return 3;
	}
	
	if(func == l_axisat)
	{
		lua_pushstring(L, "Return the site, easy axis and strength at the given index.");
		lua_pushstring(L, "1 Integer: Index of the axis.");
		lua_pushstring(L, "1 Table of 3 Integers, 3 Tables of 3 Nubmers, 3 Numbers: Coordinates of the site, directions of the cubic axes and values of the 4th, 6th and 8th order phenomenological constants.");
		return 3;	
	}
	
	if(func == l_numofax)
	{
		lua_pushstring(L, "Return the number of easy axes in the operator");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Number of easy axes.");
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
const luaL_Reg* AnisotropyCubic::luaMethods()
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



