#include "interpolatingfunction2d.h"
#include <algorithm>
#include <iostream>
#include <deque>
#include <math.h>

InterpolatingFunction2D::InterpolatingFunction2D()
{
	data = 0;
	compiled = false;
	hasInvalidValue = false;
	refcount = 0;
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

	nx = (int)(xrange / xstep) + 1;
	ny = (int)(yrange / ystep) + 1;
	
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

	for(int i=0; i<nx*ny; i++)
	{
		if(!gotdata[i])
		{
			cerr << "Interpolate2D error. Interpolation grid is not dense, 1 or more points are missing." << endl;
			delete [] gotdata;
			delete [] data;
			data = 0;
			compiled = false;
			return false;
		}
	}
	delete [] gotdata;

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












InterpolatingFunction2D* checkInterpolatingFunction2D(lua_State* L, int idx)
{
	InterpolatingFunction2D** pp = (InterpolatingFunction2D**)luaL_checkudata(L, idx, "MERCER.interpolate2d");
    luaL_argcheck(L, pp != NULL, 1, "Interpolate2D' expected");
    return *pp;
}

static int l_if_new(lua_State* L)
{
	InterpolatingFunction2D* in = new InterpolatingFunction2D;
	
	in->refcount++;
	
	InterpolatingFunction2D** pp = (InterpolatingFunction2D**)lua_newuserdata(L, sizeof(InterpolatingFunction2D**));
	
	*pp = in;
	luaL_getmetatable(L, "MERCER.interpolate2d");
	lua_setmetatable(L, -2);
	return 1;
}

static int l_if_gc(lua_State* L)
{
	InterpolatingFunction2D* in = checkInterpolatingFunction2D(L, 1);
	if(!in) return 0;
	
	in->refcount--;
	if(in->refcount == 0)
		delete in;
	
	return 0;
}

static int l_if_adddata(lua_State* L)
{
	InterpolatingFunction2D* in = checkInterpolatingFunction2D(L, 1);
	if(!in) return 0;

	in->addData(lua_tonumber(L, 2), lua_tonumber(L, 3), lua_tonumber(L, 4));
	return 0;
}

static int l_if_compile(lua_State* L)
{
	InterpolatingFunction2D* in = checkInterpolatingFunction2D(L, 1);
	if(!in) return 0;

	in->compile();
	return 0;
}

static int l_if_value(lua_State* L)
{
	InterpolatingFunction2D* in = checkInterpolatingFunction2D(L, 1);
	if(!in) return 0;

	double d;
	if(in->getValue(lua_tonumber(L, 2), lua_tonumber(L, 3), &d))
	{
		lua_pushnumber(L, d);
		return 1;
	}

	return luaL_error(L, "Empty interpolator or data out of range");
}


static int l_if_range(lua_State* L)
{
	InterpolatingFunction2D* in = checkInterpolatingFunction2D(L, 1);
	if(!in) return 0;

	if(!in->compiled)
		in->compile();

	lua_pushnumber(L, in->xmin);
	lua_pushnumber(L, in->xmax);

	lua_pushnumber(L, in->ymin);
	lua_pushnumber(L, in->ymax);
	
	return 4;
}

static int l_if_setinvalidrange(lua_State* L)
{
	InterpolatingFunction2D* in = checkInterpolatingFunction2D(L, 1);
	if(!in) return 0;
	
	in->setInvalidValue(lua_tonumber(L, 2));
	return 0;
}

void registerInterpolatingFunction2D(lua_State* L)
{
	static const struct luaL_reg methods [] = { //methods
		{"__gc",         l_if_gc},
		{"addData",      l_if_adddata},
		{"value",        l_if_value},
		{"compile",      l_if_compile},
		{"validRange",   l_if_range},
		{"setInvalidValue", l_if_setinvalidrange},
		{NULL, NULL}
	};
		
	luaL_newmetatable(L, "MERCER.interpolate2d");
	lua_pushstring(L, "__index");
	lua_pushvalue(L, -2);  /* pushes the metatable */
	lua_settable(L, -3);  /* metatable.__index = metatable */
	luaL_register(L, NULL, methods);
	lua_pop(L,1); //metatable is registered
		
	static const struct luaL_reg functions [] = {
		{"new",                 l_if_new},
		{NULL, NULL}
	};
		
	luaL_register(L, "Interpolate2D", functions);
	lua_pop(L,1);	
}
