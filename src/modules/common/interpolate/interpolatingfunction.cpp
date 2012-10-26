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

#include "interpolatingfunction.h"
#include <algorithm>
#include <iostream>
#include <deque>

InterpolatingFunction::_node::_node(double x1, double y1, double x2, double y2)
{
	x[0] = x1;
	x[1] = x2;

	y[0] = y1;
	y[1] = y2;

	c[0] = 0;
	c[1] = 0;

	m = (y[1]-y[0])/(x[1]-x[0]);

}

InterpolatingFunction::_node::_node(_node* c0, _node* c1)
{
	x[0] = c0->x[0];
	x[1] = c1->x[1];

	y[0] = c0->y[0];
	y[1] = c1->y[1];

	m = (y[1]-y[0])/(x[1]-x[0]);

	cut = c0->x[1];

	c[0] = c0;
	c[1] = c1;
}

InterpolatingFunction::_node::~_node()
{
	if(c[0]) delete c[0];
	if(c[1]) delete c[1];
}


#define EPSILON 1E-10
bool InterpolatingFunction::_node::inrange(const double test)
{
	return (test >= x[0]-EPSILON) && (test <= x[1]+EPSILON);
}


InterpolatingFunction::InterpolatingFunction()
	: LuaBaseObject(hash32(InterpolatingFunction::typeName()))
{
	root = 0;
	compiled = false;
}


int l_adddata(lua_State* L)
{
	LUA_PREAMBLE(InterpolatingFunction, in, 1);

	if(lua_istable(L, 2))
	{
		lua_pushnil(L);
		while(lua_next(L, 2))
		{
			double k,v;

			if(lua_istable(L, -1))
			{
				lua_pushinteger(L, 1);
				lua_gettable(L, -2);
				k = lua_tonumber(L, -1);
				lua_pop(L, 1);
				
				lua_pushinteger(L, 2);
				lua_gettable(L, -2);
				v = lua_tonumber(L, -1);
				lua_pop(L, 1);
			}

			in->addData(k, v);
			lua_pop(L, 1);
		}
	}
	else
	{
		in->addData(lua_tonumber(L, -2), lua_tonumber(L, -1));
	}
	return 0;
}

int InterpolatingFunction::luaInit(lua_State* L)
{
	if(lua_istable(L, -1))
	{
		lua_pushcfunction(L, l_adddata);
		luaT_push<InterpolatingFunction>(L, this);
		lua_pushvalue(L, -3);
		lua_call(L, 2, 0);
	}

	return 0;
}

void InterpolatingFunction::push(lua_State* L)
{
	luaT_push<InterpolatingFunction>(L, this);
}
	
InterpolatingFunction::~InterpolatingFunction()
{
	if(root)
		delete root;
}

void InterpolatingFunction::addData(const double in, const double out)
{
	rawdata.push_back(pair<double,double>(in,out));
	compiled = false;
}

static bool _rawsort(const pair<double,double>& d1, const pair<double,double>& d2)
{
	return d1.first < d2.first;
}

void InterpolatingFunction::compile()
{
	sort(rawdata.begin(), rawdata.end(), _rawsort);

	if(root)
		delete root;

	deque <_node*> dq1;
	deque <_node*> dq2;

	for(unsigned int i=0; i<rawdata.size()-1; i++)
	{
		dq1.push_front(new _node(
				rawdata[i  ].first,rawdata[i  ].second,
				rawdata[i+1].first,rawdata[i+1].second));
	}

	deque <_node*>& q1 = dq1;
	deque <_node*>& q2 = dq2;

	while(q1.size() > 1)
	{
		while(!q1.empty())
		{
			if(q1.size() == 3)
			{
				_node* n1 = q1.back(); q1.pop_back();
				_node* n2 = q1.back(); q1.pop_back();
				_node* n3 = q1.back(); q1.pop_back();

				q2.push_front(new _node( new _node(n1, n2), n3));
			}
			else
			{
				_node* n1 = q1.back(); q1.pop_back();
				_node* n2 = q1.back(); q1.pop_back();

				q2.push_front(new _node(n1, n2));
			}
		}

		q1.swap(q2);
	}

	root = q1.front();
	compiled = true;
}

double InterpolatingFunction::maxX()
{
	if(!compiled)
		compile();
	_node* t = root;
	while(t->c[1])
		t = t->c[1];
	return t->x[1];
}

double InterpolatingFunction::minX()
{
	if(!compiled)
		compile();
	_node* t = root;
	while(t->c[0])
		t = t->c[0];
	return t->x[0];
}


// #include <stdio.h>
bool InterpolatingFunction::getValue(double in, double* out)
{
	if(!compiled)
		compile();

// 	if(!root || !root->inrange(in))
// 	{
// 		*out = 0;
// 		return false;
// 	}

	_node* t = root;

	while(t->c[0])
	{
		if(in < t->cut)
			t = t->c[0];
		else
			t = t->c[1];
	}
	
	if(in <= t->x[0])
	{
		*out = t->y[0];
		return true;
	}

	if(in >= t->x[1])
	{
		*out = t->y[1];
		return true;
	}
// 	printf("\n     %g between %g %g", in, t->x[0], t->x[1]);
// 	printf("[%g:%g] [%g:%g] m=%g\n", t->x[0], t->y[0], t->x[1], t->y[1], t->m);

	*out = (in - t->x[0]) * t->m + t->y[0];
	return true;
}

void InterpolatingFunction::encode(buffer* b)
{
	encodeInteger( rawdata.size(), b);
	for(unsigned int i=0; i<rawdata.size(); i++)
	{
		encodeDouble(rawdata[i].first, b);
		encodeDouble(rawdata[i].second, b);
	}
	
}

int  InterpolatingFunction::decode(buffer* b)
{
	int size = decodeInteger(b);
	if(root)
		delete root;
	rawdata.clear();
	for(int i=0; i<size; i++)
	{
		const double x = decodeDouble(b);
		const double y = decodeDouble(b);

		addData(x, y);
	}
	compile();
	return 0;
}













int l_value(lua_State* L)
{
	LUA_PREAMBLE(InterpolatingFunction, in, 1);
	double d;
	if(in->getValue(lua_tonumber(L, 2), &d))
	{
		lua_pushnumber(L, d);
		return 1;
	}

	return luaL_error(L, "Empty interpolator");
}

int l_minx(lua_State* L)
{
	LUA_PREAMBLE(InterpolatingFunction, in, 1);
	lua_pushnumber(L, in->minX());
	return 1;
}
int l_maxx(lua_State* L)
{
	LUA_PREAMBLE(InterpolatingFunction, in, 1);
	lua_pushnumber(L, in->maxX());
	return 1;
}

#if 0
static int l_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.interpolate");
	return 1;
}


#endif

int InterpolatingFunction::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Interpolate creates a 1D linear interpolating function");
		lua_pushstring(L, "1 Optional Table of Tables: A table of pairs can be passed into the .new function. Each pair will be effectively passed to the addData function."); //input, empty
		lua_pushstring(L, ""); //output, empty
		return 3;
	}
	
	if(lua_istable(L, 1))
	{
		return 0;
	}
	
	if(!lua_iscfunction(L, 1))
	{
		return luaL_error(L, "help expect zero arguments or 1 function");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if(func == l_adddata)
	{
		lua_pushstring(L, "Add data to the 1D linear interpolator.");
		lua_pushstring(L, "2 numbers: the x and y values where x is the data position and y is the data value.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_value)
	{
		lua_pushstring(L, "Interpolate a value from the 1D linear interpolator.");
		lua_pushstring(L, "1 number: the x value or data position which will have a data value interpolated.");
		lua_pushstring(L, "1 number: the interpolated data value at the input position.");
		return 3;
	}
	
	if(func == l_maxx)
	{
		lua_pushstring(L, "Get maximum X value.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: Maximum X value");
		return 3;
	}	
	
	if(func == l_minx)
	{
		lua_pushstring(L, "Get minimum X value.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Number: Minimum X value");
		return 3;
	}	
	
	
	return LuaBaseObject::help(L);
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* InterpolatingFunction::luaMethods()
{
	if(m[127].name)return m;

	static const luaL_Reg _m[] =
	{
		{"addData",      l_adddata},
		{"value",        l_value},
		{"maxX", l_maxx},
		{"minX", l_minx},
		{"__call",        l_value},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}







#include "info.h"
#include "interpolatingfunction2d.h"
extern "C"
{
INTERPOLATE_API int lib_register(lua_State* L);
INTERPOLATE_API int lib_version(lua_State* L);
INTERPOLATE_API const char* lib_name(lua_State* L);
INTERPOLATE_API int lib_main(lua_State* L);
}

INTERPOLATE_API int lib_register(lua_State* L)
{
	luaT_register<InterpolatingFunction>(L);
	luaT_register<InterpolatingFunction2D>(L);
	return 0;
}

INTERPOLATE_API int lib_version(lua_State* L)
{
	return __revi;
}

INTERPOLATE_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Interpolate";
#else
	return "Interpolate-Debug";
#endif
}

INTERPOLATE_API int lib_main(lua_State* L)
{
	return 0;
}
