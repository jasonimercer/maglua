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












InterpolatingFunction* checkInterpolatingFunction(lua_State* L, int idx)
{
	InterpolatingFunction** pp = (InterpolatingFunction**)luaL_checkudata(L, idx, "MERCER.interpolate");
    luaL_argcheck(L, pp != NULL, 1, "Interpolate' expected");
    return *pp;
}

int l_if_new(lua_State* L)
{
	InterpolatingFunction* in = new InterpolatingFunction;
	
	in->refcount++;
	
	InterpolatingFunction** pp = (InterpolatingFunction**)lua_newuserdata(L, sizeof(InterpolatingFunction**));
	
	*pp = in;
	luaL_getmetatable(L, "MERCER.interpolate");
	lua_setmetatable(L, -2);
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

void registerInterpolatingFunction(lua_State* L)
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
		{NULL, NULL}
	};
		
	luaL_register(L, "Interpolate", functions);
	lua_pop(L,1);	
}
