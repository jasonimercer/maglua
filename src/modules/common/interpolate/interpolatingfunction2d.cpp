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

#include "interpolatingfunction2d.h"
#include <algorithm>
#include <iostream>
#include <deque>
#include <math.h>

InterpolatingFunction2D::InterpolatingFunction2D()
	: LuaBaseObject(hash32(InterpolatingFunction2D::typeName()))
{
	data = 0;
	compiled = false;
	hasInvalidValue = false;
}

int InterpolatingFunction2D::luaInit(lua_State* L)
{
	return 0;
}
	
	
InterpolatingFunction2D::~InterpolatingFunction2D()
{
	if(data)
		delete [] data;
}

void InterpolatingFunction2D::addData(const double inx, const double iny, const double out)
{
	rawdata.push_back( triple(inx, iny, out) );
	compiled = false;
}

bool InterpolatingFunction2D::compile()
{
	//try to figure out max, min and steps
	xmin = rawdata[0].x;
	xmax = rawdata[0].x;
	ymin = rawdata[0].y;
	ymax = rawdata[0].y;

	for(unsigned int i=1; i<rawdata.size(); i++)
	{
		if(rawdata[i].x < xmin) xmin = rawdata[i].x;
		if(rawdata[i].x > xmax) xmax = rawdata[i].x;
		if(rawdata[i].y < ymin) ymin = rawdata[i].y;
		if(rawdata[i].y > ymax) ymax = rawdata[i].y;
	}

	double xrange = xmax - xmin;
	double yrange = ymax - ymin;

	if(xrange == 0 || yrange == 0)
		return false;

	xstep = xrange;
	ystep = yrange;

	//looking for the smallest difference in coords 
	//that's greater than range / 1e7
	for(unsigned int i=1; i<rawdata.size(); i++)
	{
		double dx = fabs(rawdata[i].x - rawdata[0].x);
		double dy = fabs(rawdata[i].y - rawdata[0].y);
		if(dx > (xrange * 1e-7) && dx < xstep)
			xstep = dx;
		if(dy > (yrange * 1e-7) && dy < ystep)
			ystep = dy;
	}

	if(xstep == 0 || ystep == 0)
		return false;

	//cout << xrange << ":" << yrange << ":" << xstep << ":" << ystep << endl;
	
	nx = (int)(xrange / xstep) + 1;
	ny = (int)(yrange / ystep) + 1;
	
	if(data)
		delete [] data;
	data = new double[nx*ny];
	bool* gotdata = new bool[nx*ny];
	
	for(int i=0; i<nx*ny; i++)
		gotdata[i] = false;


	for(unsigned int i=0; i<rawdata.size(); i++)
	{
		triple& t = rawdata[i];
		int idx = getidx(t.x, t.y);

		if(idx < 0 || idx >= nx*ny)
		{
			cerr << "Interpolate2D data (" << t.x << ", " << t.y << ", " << t.z << ") is out of range" << endl;
			delete [] gotdata;
			delete [] data;
			data = 0;
			compiled = false;
			return false;
		}
		data[idx] = t.z;
		gotdata[idx] = true;
	}

	int missingpts = 0;
	for(int i=0; i<nx*ny; i++)
	{
		if(!gotdata[i])
		{
			missingpts++;
		}
	}
	
	
	delete [] gotdata;
	
	if(missingpts)
	{
		cerr << "Interpolate2D error. Interpolation grid is not dense, " << missingpts << 
			"/" << nx*ny << " point" <<(missingpts!=1?"s are":" is") << " missing." << endl;
		delete [] data;
		data = 0;
		compiled = false;
		return false;
	}
	
	
	compiled = true;
	return true;
}



int InterpolatingFunction2D::getidx(double x, double y)
{
	int v2[2];
	getixiy(x, y, v2);

	return v2[0] + v2[1] * nx;
}

void InterpolatingFunction2D::getixiy(double x, double y, int* v2)
{
	v2[0] = (int)floor( (x-xmin) / xstep );
	v2[1] = (int)floor( (y-ymin) / ystep );
}

void InterpolatingFunction2D::setInvalidValue(const double d)
{
	hasInvalidValue = true;
	invalidValue = d;
}


bool InterpolatingFunction2D::getValue(double x, double y, double* z)
{
	int ixy[2];
	if(!compiled)
		compile();
	
	getixiy(x, y, ixy);
	
	if(ixy[0] < 0 || ixy[0] >= nx || ixy[1] < 0 || ixy[1] >= ny)
	{
		if(hasInvalidValue)
		{
			*z = invalidValue;
			return true;
		}
		return false;
	}

	const double t = (x - (ixy[0] * xstep + xmin)) / xstep;
	const double u = (y - (ixy[1] * ystep + ymin)) / ystep;

	const int ix = ixy[0];
	const int iy = ixy[1];

	int ixp = ix + 1;
	int iyp = iy + 1;
	
	if(ixp >= nx) ixp = nx -1;
	if(iyp >= ny) iyp = ny -1;
	
	const double z0 = data[ (ix ) + (iy )*nx ];
	const double z1 = data[ (ixp) + (iy )*nx ];
	const double z2 = data[ (ixp) + (iyp)*nx ];
	const double z3 = data[ (ix ) + (iyp)*nx ];

	*z = (1.0 - t)*(1.0 - u) * z0 +
		 (      t)*(1.0 - u) * z1 +
		 (      t)*(      u) * z2 +
		 (1.0 - t)*(      u) * z3;
	return true;
}

void InterpolatingFunction2D::encode(buffer* b)
{
	ENCODE_PREAMBLE
	encodeInteger( rawdata.size(), b);
	for(unsigned int i=0; i<rawdata.size(); i++)
	{
		const triple& t = rawdata[i];
		
		//cout << "encode (" << i << ", " << t.x << ", " << t.y << ", " << t.z << ")" << endl;
		encodeDouble(t.x, b);
		encodeDouble(t.y, b);
		encodeDouble(t.z, b);
	}
}

int  InterpolatingFunction2D::decode(buffer* b)
{
	char version = decodeChar(b);
	if(version == 0)
	{
		int size = decodeInteger(b);
		rawdata.clear();
		for(int i=0; i<size; i++)
		{
			const double x = decodeDouble(b);
			const double y = decodeDouble(b);
			const double z = decodeDouble(b);
			//cout << "decode (" << i << ", " << x << ", " << y << ", " << z << ")" << endl;
			addData(x, y, z);
		}
	}
	else
	{
		fprintf(stderr, "(%s:%i) %s::decode, unknown version:%i\n", __FILE__, __LINE__, lineage(0), (int)version);
	}

	return 0;
}











static int l_adddata(lua_State* L)
{
	LUA_PREAMBLE(InterpolatingFunction2D, in, 1);
	in->addData(lua_tonumber(L, 2), lua_tonumber(L, 3), lua_tonumber(L, 4));
	return 0;
}

static int l_compile(lua_State* L)
{
	LUA_PREAMBLE(InterpolatingFunction2D, in, 1);
	in->compile();
	return 0;
}

static int l_value(lua_State* L)
{
	LUA_PREAMBLE(InterpolatingFunction2D, in, 1);
	double d;
	if(in->getValue(lua_tonumber(L, 2), lua_tonumber(L, 3), &d))
	{
		lua_pushnumber(L, d);
		return 1;
	}

	return luaL_error(L, "Empty interpolator or data out of range");
}


static int l_range(lua_State* L)
{
	LUA_PREAMBLE(InterpolatingFunction2D, in, 1);
	if(!in->compiled)
		in->compile();

	lua_pushnumber(L, in->xmin);
	lua_pushnumber(L, in->xmax);

	lua_pushnumber(L, in->ymin);
	lua_pushnumber(L, in->ymax);
	
	return 4;
}

static int l_setinvalidrange(lua_State* L)
{
	LUA_PREAMBLE(InterpolatingFunction2D, in, 1);
	in->setInvalidValue(lua_tonumber(L, 2));
	return 0;
}

#if 0
static int l_mt(lua_State* L)
{
	luaL_getmetatable(L, "MERCER.interpolate2d");
	return 1;
}

static int l_help(lua_State* L)
{

	
	return 0;
}
#endif


int InterpolatingFunction2D::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Interpolate2D creates a 2D linear interpolating function. Data must form a complete grid.");
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
		return luaL_error(L, "help expect zero arguments or 1 function.");
	}
	
	lua_CFunction func = lua_tocfunction(L, 1);
	
	if(func == l_adddata)
	{
		lua_pushstring(L, "Add data to the 2D linear interpolator.");
		lua_pushstring(L, "3 numbers: the x, y and z values where (x, y) is the data position and z is the data value.");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_value)
	{
		lua_pushstring(L, "Interpolate a value from the 2D linear interpolator.");
		lua_pushstring(L, "2 numbers: the x and y value representing the data position which will have a value interpolated.");
		lua_pushstring(L, "1 number: the interpolated data value at the input position.");
		return 3;
	}
	
	if(func == l_compile)
	{
		lua_pushstring(L, "Compute all internal variables needed to interpolate a 2D value as well as data range. If this method is not called, it is done automatically when the first interpolated value is requested");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}
	
	if(func == l_range)
	{
		lua_pushstring(L, "Compile the interpolating function if needed and return the valid data range of the interpolator.");
		lua_pushstring(L, "");
		lua_pushstring(L, "4 numbers: The xmin, xmax, ymin, ymax values");
		return 3;
	}
	
	if(func == l_setinvalidrange)
	{
		lua_pushstring(L, "Set the value to return if a requested data point is outside the data range.");
		lua_pushstring(L, "1 number: Value of data outside data range");
		lua_pushstring(L, "");
		return 3;
	}
	
	return LuaBaseObject::help(L);
}


static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* InterpolatingFunction2D::luaMethods()
{
	if(m[127].name)return m;

	static const luaL_Reg _m[] =
	{
		{"addData",      l_adddata},
		{"value",        l_value},
		{"compile",      l_compile},
		{"validRange",   l_range},
		{"setInvalidValue", l_setinvalidrange},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}
