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
	: Encodable(ENCODE_INTERP1D)
{
	root = 0;
	compiled = false;
	refcount = 0;
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

// #include <stdio.h>
bool InterpolatingFunction::getValue(double in, double* out)
{
	if(!compiled)
		compile();

	if(!root || !root->inrange(in))
	{
		*out = 0;
		return false;
	}

	_node* t = root;

	while(t->c[0])
	{
		if(in < t->cut)
			t = t->c[0];
		else
			t = t->c[1];
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












InterpolatingFunction* checkInterpolatingFunction(lua_State* L, int idx)
{
	InterpolatingFunction** pp = (InterpolatingFunction**)luaL_checkudata(L, idx, "MERCER.interpolate");
    luaL_argcheck(L, pp != NULL, 1, "Interpolate' expected");
    return *pp;
}

void lua_pushInterpolatingFunction(lua_State* L, Encodable* _if1D)
{
	InterpolatingFunction* if1D = dynamic_cast<InterpolatingFunction*>(_if1D);
	if(!if1D) return;
	if1D->refcount++;
	
	InterpolatingFunction** pp = (InterpolatingFunction**)lua_newuserdata(L, sizeof(InterpolatingFunction**));
	
	*pp = if1D;
	luaL_getmetatable(L, "MERCER.interpolate");
	lua_setmetatable(L, -2);
}

int l_if_new(lua_State* L)
{
	lua_pushInterpolatingFunction(L, new InterpolatingFunction);
	return 1;
}

int l_if_gc(lua_State* L)
{
	InterpolatingFunction* in = checkInterpolatingFunction(L, 1);
	if(!in) return 0;
	
	in->refcount--;
	if(in->refcount == 0)
		delete in;
	
	return 0;
}

int l_if_adddata(lua_State* L)
{
	InterpolatingFunction* in = checkInterpolatingFunction(L, 1);
	if(!in) return 0;

	in->addData(lua_tonumber(L, -2), lua_tonumber(L, -1));
	return 0;
}

int l_if_value(lua_State* L)
{
	InterpolatingFunction* in = checkInterpolatingFunction(L, 1);
	if(!in) return 0;

	double d;
	if(in->getValue(lua_tonumber(L, 2), &d))
	{
		lua_pushnumber(L, d);
		return 1;
	}

	return luaL_error(L, "Empty interpolator or data out of range");
}


static int l_if_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.interpolate");
	return 1;
}

static int l_if_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Interpolate creates a 1D linear interpolating function");
		lua_pushstring(L, ""); //input, empty
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
	
	if(func == l_if_new)
	{
		lua_pushstring(L, "Create a new Interpolate object.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Interpolate object");
		return 3;
	}
	
	if(func == l_if_adddata)
	{
		lua_pushstring(L, "Add data to the 1D linear interpolator.");
		lua_pushstring(L, "2 numbers: the x and y values where x is the data position and y is the data value.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_if_value)
	{
		lua_pushstring(L, "Interpolate a value from the 1D linear interpolator.");
		lua_pushstring(L, "1 number: the x value or data position which will have a data value interpolated.");
		lua_pushstring(L, "1 number: the interpolated data value at the input position.");
		return 3;
	}
	
	return 0;
}

static Encodable* newThing()
{
	return new InterpolatingFunction;
}

int registerInterpolatingFunction(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_if_gc},
		{"addData",      l_if_adddata},
		{"value",        l_if_value},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.interpolate");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_if_new},
		{"help",                l_if_help},
		{"metatable",           l_if_mt},
		{NULL, NULL}
	};
		
	luaL_register(L, "Interpolate", functions);
	lua_pop(L,1);	
	
	return Factory_registerItem(ENCODE_INTERP1D, newThing, lua_pushInterpolatingFunction, "Interpolate");
}








#include "info.h"
#include "interpolatingfunction2d.h"
extern "C"
{
INTERPOLATE_API int lib_register(lua_State* L);
INTERPOLATE_API int lib_version(lua_State* L);
INTERPOLATE_API const char* lib_name(lua_State* L);
INTERPOLATE_API int lib_main(lua_State* L, int argc, char** argv);
}

INTERPOLATE_API int lib_register(lua_State* L)
{
	const int a = registerInterpolatingFunction(L);
	const int b = registerInterpolatingFunction2D(L);
	return a | b;
}

INTERPOLATE_API int lib_version(lua_State* L)
{
	return __revi;
}

INTERPOLATE_API const char* lib_name(lua_State* L)
{
	return "Interpolate";
}

INTERPOLATE_API int lib_main(lua_State* L, int argc, char** argv)
{
	return 0;
}
