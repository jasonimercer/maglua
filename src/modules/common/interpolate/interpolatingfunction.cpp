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

void InterpolatingFunction::clear()
{
    rawdata.clear();
    compiled = false;
}

InterpolatingFunction* InterpolatingFunction::inverted()
{
	InterpolatingFunction* i = new InterpolatingFunction;
	i->L = L;
	vector <pair<double,double> >::iterator it;
	for(it=rawdata.begin(); it != rawdata.end(); ++it)
	{
		i->addData( (*it).second, (*it).first );
	}

	return i;
}


static bool _rawsort(const pair<double,double>& d1, const pair<double,double>& d2)
{
	return d1.first < d2.first;
}

static bool removeSameX(vector <pair<double,double> >& rawdata)
{
	vector <pair<double,double> >::iterator it1;
	vector <pair<double,double> >::iterator it2;
	
	if(rawdata.size() <= 1)
		return false; //done
		
	it1 = rawdata.begin();
	it2 = rawdata.begin();
	it2++;
	
	for(;it2 != rawdata.end(); it1++, it2++)
	{
		if( (*it1).first == (*it2).first) 
		{
			rawdata.erase(it2);
			return true;
		}
	}

	return false;
}


void InterpolatingFunction::compile()
{
	sort(rawdata.begin(), rawdata.end(), _rawsort);

	while(removeSameX(rawdata));
	
	if(root)
		delete root;

	if(rawdata.size() == 0)
	{
		compiled = true;
		root = new _node(0,0,0,0);
		return;
	}
	if(rawdata.size() == 1)
	{
		compiled = true;
		root = new _node(rawdata[0].first,rawdata[0].second,
						 rawdata[0].first,rawdata[0].second);
		return;
	}
	
	deque <_node*> q1;
	deque <_node*> q2;

	for(unsigned int i=0; i<rawdata.size()-1; i++)
	{
		q1.push_front(new _node(
				rawdata[i  ].first,rawdata[i  ].second,
				rawdata[i+1].first,rawdata[i+1].second));
	}

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


bool InterpolatingFunction::getValue(double in, double& out)
{
    return getValue(in, &out);
}

// #include <stdio.h>
bool InterpolatingFunction::getValue(double in, double* out)
{
	if(!compiled)
		compile();

	_node* t = root;
	
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
	ENCODE_PREAMBLE
	char version = 0;
	
	encodeChar(version, b);
	encodeInteger( rawdata.size(), b);
	for(unsigned int i=0; i<rawdata.size(); i++)
	{
		encodeDouble(rawdata[i].first, b);
		encodeDouble(rawdata[i].second, b);
	}
	
}

int  InterpolatingFunction::decode(buffer* b)
{
	char version = decodeChar(b);
	if(version == 0)
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
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
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

int l_clear(lua_State* L)
{
	LUA_PREAMBLE(InterpolatingFunction, in, 1);
        in->clear();
	return 0;
}

int l_td(lua_State* L)
{
    LUA_PREAMBLE(InterpolatingFunction, in, 1);

    
    vector <pair<double,double> >::iterator it;

    lua_newtable(L);
    for(it=in->rawdata.begin(); it != in->rawdata.end(); ++it)
    {
        lua_pushnumber(L, it->first);
        lua_pushnumber(L, it->second);
        lua_settable(L, -3);
    }
    
    return 1;
}

int l_inverted(lua_State* L)
{
	LUA_PREAMBLE(InterpolatingFunction, in, 1);

	luaT_push<InterpolatingFunction>(L, in->inverted());
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
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if(func == l_adddata)
	{
		lua_pushstring(L, "Add data to the 1D linear interpolator.");
		lua_pushstring(L, "2 numbers or a table of pairs of numbers: the x and y values where x is the data position and y is the data value.");
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
	if(func == l_clear)
	{
		lua_pushstring(L, "Clears the interpolator");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}	
	
        if(func == l_td)
        {
            lua_pushstring(L, "Get the key-value pairs that define this interpolator");
            lua_pushstring(L, "");
            lua_pushstring(L, "1 Table: Keys are the x values, values are the y values.");
            
        }

	if(func == l_inverted)
	{
		lua_pushstring(L, "Return a new 1D linear interpolating function based on the calling interpolator with the data and values interchanged. The calling interpolator should be monotonic.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 1D Interpolator: Inverted interpolator");
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
		{"inverted",     l_inverted},
		{"maxX", l_maxx},
		{"minX", l_minx},
                {"clear", l_clear},
                {"tableData", l_td},
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
