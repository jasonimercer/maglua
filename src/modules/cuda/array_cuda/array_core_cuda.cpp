#include "array_core_cuda.h"

template<typename T>
static int l_mattrans(lua_State* L);

static int l_matdet(lua_State* L);

static int l_matmul(lua_State* L);

template<typename T>
static int l_sameSize(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	LUA_PREAMBLE(Array<T>, b, 2);
	lua_pushboolean(L, a->sameSize(b));
	return 1;
}

template<typename T>
static int l_get_nx(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	lua_pushinteger(L, a->nx);
	return 1;
}
template<typename T>
static int l_get_ny(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	lua_pushinteger(L, a->ny);
	return 1;
}
template<typename T>
static int l_get_nz(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	lua_pushinteger(L, a->nz);
	return 1;
}
template<typename T>
static int l_get(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	return a->lua_get(L, 2);
}

const char* table_set_code = 
"return function(A, t)\n"
"	local dims = 0\n"
"	local tt = \"table\"\n"
"	if t and type(t) == tt then\n"
"		dims = 1\n"
"		if t[1] and type(t[1]) == tt then\n"
"			dims = 2\n"
"			if t[1][1] and type(t[1][1]) == tt then\n"
"				dims = 3\n"
"			end\n"
"		end\n"
"	end\n"
"	\n"
"	for i=dims,2 do --promote to dims = 3\n"
"		t = {t}\n"
"	end\n"
"	\n"
"	for z,v1 in ipairs(t) do\n"
"		for y,v2 in ipairs(v1) do\n"
"			for x,v in ipairs(v2) do\n"
"				A:set({x,y,z}, v)\n"
"			end\n"
"		end\n"
"	end\n"
"end\n";

template<typename T>
static int l_set(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	
	if(luaT_is< Array<T> >(L, lua_gettop(L)))
	{
		LUA_PREAMBLE(Array<T>, b, lua_gettop(L));
		a->copyFrom(b);
		return 0;		
	}
	
	return a->lua_set(L, 2);
}

template<typename T>
static int l_copy(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	
	Array<T>* b = 0;
	if(luaT_is< Array<T> >(L, 2))
	{
		b = luaT_to< Array<T> >(L, 2);
		if(!b->sameSize(a))
		{
			return luaL_error(L, "Destination array size mismatch");
		}
	}
	else
		b = new Array<T>(a->nx, a->ny, a->nz);
	
	b->copyFrom(a);
	luaT_push< Array<T> >(L, b);
	
	return 1;
}

template<typename T>
static int l_addat(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	return a->lua_addat(L, 2);
}

template<typename T>
static int l_setfromtable(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
		
	if(lua_gettop(L) == 2 && lua_istable(L, 2))
	{
		luaL_dostring(L, table_set_code);
		lua_pushvalue(L, 1);
		lua_pushvalue(L, 2);
		lua_call(L, 2, 0);
	}
	else
	{
		return luaL_error(L, "Expected a single table");
	}

	return 0;
}

template<typename T>
static int l_setAll(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	T t = luaT<T>::to(L, 2);
	a->setAll(t);
	return 0;
}
template<typename T>
static int l_zero(lua_State* L)
{
 	LUA_PREAMBLE(Array<T>, a, 1);
	T t = luaT<T>::zero();
	a->setAll(t);
	return 0;
}
template<typename T>
static int l_pwm(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	LUA_PREAMBLE(Array<T>, b, 2);
	LUA_PREAMBLE(Array<T>, c, 3);
	
	Array<T>::pairwiseMult(c, a, b);
	return 0;
}
template<typename T>
static int l_pwsa(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);

	const int k = luaT<T>::elements()-1;
	
	if(!luaT<T>::is(L, 2))
		return luaL_error(L, "expected a scaling value");
	T scale = luaT<T>::to(L, 2);
	
	LUA_PREAMBLE(Array<T>, b, 3+k);
	
	Array<T>* c = 0;
	if(luaT_is< Array<T> >(L, 4+k))
		c = luaT_to< Array<T> >(L, 4+k);
	else
		c = new Array<T>(a->nx, a->ny, a->nz);
	
	luaT_push< Array<T> >(L, c);
	
	if(!Array<T>::pairwiseScaleAdd(c, luaT<T>::one(), a, scale, b))
		return luaL_error(L, "Failed to pairwiseScaleAdd");
	return 1;
}

template<typename T>
static int l_dot(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	LUA_PREAMBLE(Array<T>, b, 2);
	
	T t = Array<T>::dot(a,b);
	luaT<T>::push(L, t);
	return 1;
}

template<typename T>
static int l_min(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	int idx;
	T t = a->min(idx);

	luaT<T>::push(L, t);
	lua_pushinteger(L, idx+1);
	return luaT<T>::elements() + 1;
}

template<typename T>
static int l_max(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	int idx;
	T t = a->max(idx);

	luaT<T>::push(L, t);
	lua_pushinteger(L, idx+1);
	return luaT<T>::elements() + 1;
}

template<typename T>
static int l_mean(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	T t = a->mean();
	luaT<T>::push(L, t);
	return luaT<T>::elements();
}
template<typename T>
static int l_sum(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	double p = 1.0;
	if(lua_isnumber(L, 2))
		p = lua_tonumber(L, 2);
	T t = a->sum(p);
	luaT<T>::push(L, t);
	return luaT<T>::elements();
}
template<typename T>
static int l_scale(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	T t = luaT<T>::to(L, 2);
	a->scaleAll(t);
	return 0;
}

static void get_region(lua_State* L, int idx, int* r6)
{
	if(!lua_istable(L, idx))
		return;
	
	for(int i=1; i<=2; i++)
	{
		lua_pushinteger(L, i);
		lua_gettable(L, idx);
		
		if(!lua_istable(L, -1))
			return;
		
		for(int j=0; j<3; j++)
		{
			lua_pushinteger(L, j+1);
			lua_gettable(L, -2);
			int k = (i-1)*3 + j;
			if(lua_isnumber(L, -1))
			{
				r6[k] = lua_tointeger(L, -1) -1;
				if(r6[k] < 0)
					r6[k] = 0;
			}
			else
			{
				r6[k] = 0;
			}
			lua_pop(L, 1); //pop number
		}
		lua_pop(L, 1); //pop table
	}
	
}

static int same_shape(int* r1, int* r2)
{
	for(int i=0; i<3; i++)
		if( (r2[i+3] - r2[i]) != (r1[i+3] - r1[i]))
			return 0;
	return 1;
}

template<typename T>
static int l_manip_region(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, src, 1);

	int nt = 0; //number of tables
	int  region[2][6] = {{0,0,0,0,0,0},{0,0,0,0,0,0}};
	
	for(int i=2; i<=lua_gettop(L) && nt < 2; i++)
	{
		if(lua_istable(L, i))
		{
			get_region(L, i, region[nt]);
			nt++;
		}
	}
	
	if(nt == 0)
	{
		return luaL_error(L, "Source region required");
	}
	
	int size[3];
	for(int i=0; i<3; i++)
		size[i] = region[0][i+3] - region[0][i] + 1;
	
	if(nt == 1) // make default destination region
	{
		for(int i=0; i<3; i++)
		{
			region[1][i]   = 0;
			region[1][i+3] = region[0][i+3] - region[0][i];
		}
	}

	if(!same_shape(region[0], region[1]))
		return luaL_error(L, "Source and destination shapes are not the same");

	Array<T>* dest = 0;
	for(int i=2; i<=lua_gettop(L) && (dest == 0); i++)
	{
		if(luaT_is< Array<T> >(L, i))
		{
			dest = luaT_to< Array<T> >(L, i);
		}
	}
	
	if(!dest)
		dest = new Array<T>(region[1][3]+1, region[1][4]+1, region[1][5]+1);
	
	luaT_push< Array<T> >(L, dest);
	
	Array<T>* aa[2];
	aa[0] = src;
	aa[1] = dest;
	
	for(int a=0; a<2; a++)
	{
		for(int k=0; k<2; k++)
			if(!aa[a]->member(region[a][0+3*k], region[a][1+3*k],region[a][2+3*k]))
				return luaL_error(L, "Region out of bounds");
	}
	
	src->copyRegionFromTo(region[0], region[0]+3, dest,  region[1], region[1]+3);
	
	return 1;
}



template<typename T>
static const luaL_Reg* get_base_methods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	static const luaL_Reg _m[] =
	{
		{"setAll",  l_setAll<T>},
		{"pairwiseMultiply",     l_pwm<T>},
		{"pairwiseScaleAdd",    l_pwsa<T>},
		{"dot",     l_dot<T>},
		{"zero",    l_zero<T>},
		{"sameSize",l_sameSize<T>},
		{"nx",      l_get_nx<T>},
		{"ny",      l_get_ny<T>},
		{"nz",      l_get_nz<T>},
		{"get",     l_get<T>},
		{"set",     l_set<T>},
		{"copy",    l_copy<T>},
		{"setFromTable", l_setfromtable<T>},
		{"addAt",   l_addat<T>},
		{"min",     l_min<T>},
		{"max",     l_max<T>},
		{"mean",    l_mean<T>},
		{"scale",   l_scale<T>},
		{"slice",    l_manip_region<T>},
		{"sum",     l_sum<T>},
		
		{"matTrans",     l_mattrans<T>},
		{"matDet",       l_matdet},
		{"matMul",       l_matmul},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}


template<typename T>
int Array<T>::help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Class for 3D Data Arrays (CUDA Implementation)");
		lua_pushstring(L, "0 to 3 Integers: Length of each dimension X, Y and Z. Default values are 1.");
		lua_pushstring(L, ""); //output, empty
		return 3;
	}

	if(!lua_iscfunction(L, 1))
	{
		return luaL_error(L, "help expects zero arguments or 1 function.");
	}


	lua_CFunction func = lua_tocfunction(L, 1);
	if(func == &(l_sameSize<T>))
	{
		lua_pushstring(L, "Test if a given array has the same dimensions");
		lua_pushstring(L, "1 Array");
		lua_pushstring(L, "1 Boolean: True if sizes match");
		return 3;
	}
	if(func == &(l_setAll<T>))
	{
		lua_pushstring(L, "Set all values in the array to the given value");
		lua_pushstring(L, "1 Value: The new value for all element entries");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == &(l_pwm<T>))
	{
		lua_pushstring(L, "Multiply each data in this array with the data in another storing in a destination array");
		lua_pushstring(L, "2 Arrays: The pairwise scaling array and the destination array");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == &(l_pwsa<T>))
	{
		lua_pushstring(L, "Add each data in this array with a constant factor times the data in another storing in a destination array");
		lua_pushstring(L, "1 Value, 1 Source Array, 1 Optional Destination Array: The factor to scale the source array and the destination of the pairwise addition");
		lua_pushstring(L, "1 Destination Array: If no destination array is supplied, it will be created");
		return 3;
	}
	if(func == &(l_dot<T>))
	{
		lua_pushstring(L, "Compute the dot product of the current array and another of equal size and type");
		lua_pushstring(L, "1 Arrays: other array");
		lua_pushstring(L, "1 Value: result of dot product");
		return 3;
	}
	
	if(func == &(l_zero<T>))
	{
		lua_pushstring(L, "Set all values in the array 0");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == &(l_get_nx<T>))
	{
		lua_pushstring(L, "Return the size of the X dimension");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size fo the X dimension");
		return 3;
	}
	if(func == &(l_get_ny<T>))
	{
		lua_pushstring(L, "Return the size of the Y dimension");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size fo the Y dimension");
		return 3;
	}
	if(func == &(l_get_nz<T>))
	{
		lua_pushstring(L, "Return the size of the Z dimension");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size fo the Z dimension");
		return 3;
	}

	if(func == &(l_get<T>))
	{
		lua_pushstring(L, "Get an element from the array");
		lua_pushstring(L, "1, 2 or 3 integers (or 1 table): indices(XYZ) of the element to fetch default values are 1");
		lua_pushstring(L, "1 value");
		return 3;
	}
	if(func == &(l_set<T>))
	{
		lua_pushstring(L, "Set an element of the array");
		lua_pushstring(L, "1, 2 or 3 integers (or 1 table), 1 value: indices(XYZ) of the element to set, default values are 1. Last argument is the new value");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == &(l_setfromtable<T>))
	{
		lua_pushstring(L, "Set elements of the array based on table values");
		lua_pushstring(L, "1 1D, 2D or 3D table of values: new values to set");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == &(l_copy<T>))
	{
		lua_pushstring(L, "Create a copy of an array");
		lua_pushstring(L, "1 Optional Array: Destination array, if it is not provided a new array will be created.");
		lua_pushstring(L, "1 Array: A copy of the array");
		return 3;
	}
	if(func == &(l_addat<T>))
	{
		lua_pushstring(L, "Add a value to an element of the array");
		lua_pushstring(L, "1, 2 or 3 integers (or 1 table), 1 value: indices(XYZ) of the element to modify, default values are 1. Last argument is the value to add");
		lua_pushstring(L, "");
		return 3;
	}
	if(func == &(l_min<T>))
	{
		lua_pushstring(L, "Find minimum value and corresponding index");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 value and 1 integer");
		return 3;
	}
	if(func == &(l_max<T>))
	{
		lua_pushstring(L, "Find maximum value and corresponding index");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 value and 1 integer");
		return 3;
	}
	if(func == &(l_mean<T>))
	{
		lua_pushstring(L, "Find mean of array");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 value");
		return 3;
	}
	if(func == &(l_manip_region<T>))
	{
		lua_pushstring(L, "Extract a region of an array or copy to another region of another array (or same array)");
		lua_pushstring(L, "1 Table of Tables, 1 Optional Array, 1 Optional Table of Tables: The first argument represents the min and max corners of the slice that will be copied from the calling Array, the coordinates can be up to 3 integers long representing Cartesian coordinates, missing dimensions will be assumed to be 1. If an Array is given then the slice will be copied into it at the provided optional location, if an Array is not supplied then a new one will be created. The last optional table is the destination region for the data in the destination table. The size of the source and destination slices must match.");
		lua_pushstring(L, "1 Array: Either the supplied Array with the data copied in or a new array.");
		return 3;
	}
	
	lua_CFunction f18 = l_mattrans<T>;
	if(func == f18)
	{
		lua_pushstring(L, "Transpose the z=1 section of the array");
		lua_pushstring(L, "1 Optional Array: Destination array which will contain the transpose, may be the calling array. Must be of the appropriate dimensions.");
		lua_pushstring(L, "1 Array: Transpose of array. If the optional array is given it will be the same array otherwaise a new array will be created.");
		return 3;
	}

	
	lua_CFunction f19 = l_matdet;
	if(func == f19)
	{
		lua_pushstring(L, "Treat array like a matrix and compute the determinant of the z=1 slice.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Value: The determinant.");
		return 3;
	}
	
	lua_CFunction f20 = l_matmul;
	if(func == f20)
	{
		lua_pushstring(L, "Treat arrays like matrices and do Matrix Multiplication on the z=1 layer");
		lua_pushstring(L, "1 Array, 1 Optional Array: The given array will be multiply the calling Array, their dimensions must match to allow legal matrix multiplication. If a 2nd Array is supplied the product will be stored in it, otherise a new Array will be created.");
		lua_pushstring(L, "1 Array: The product of the multiplication.");
		return 3;
	}
	
	
	if(func == &(l_scale<T>))
	{
		lua_pushstring(L, "Scale all values in the array by the given value");
		lua_pushstring(L, "1 Value: The scaling factor");
		lua_pushstring(L, "");
		return 3;
	}
	return LuaBaseObject::help(L);
}








template<typename T>
static int l_fft1D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	a->fft1DTo(b);
	return 0;
}
template<typename T>
static int l_fft2D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	a->fft2DTo(b);
	return 0;
}
template<typename T>
static int l_fft3D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	a->fft3DTo(b);
	return 0;
}



template<typename T>
static int l_ifft1D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	a->ifft1DTo(b);
	return 0;
}
template<typename T>
static int l_ifft2D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	a->ifft2DTo(b);
	return 0;
}
template<typename T>
static int l_ifft3D(lua_State *L)
{
	LUA_PREAMBLE( Array<T>, a, 1);
	LUA_PREAMBLE( Array<T>, b, 2);
	a->ifft3DTo(b);
	return 0;
}


template<typename T>
static const luaL_Reg* get_fft_methods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	static const luaL_Reg _m[] =
	{
		{"fft1D",   l_fft1D<T>},
		{"fft2D",   l_fft2D<T>},
		{"fft3D",   l_fft3D<T>},

		{"ifft1D",  l_ifft1D<T>},
		{"ifft2D",  l_ifft2D<T>},
		{"ifft3D",  l_ifft3D<T>},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}






// the level argument below prevents WSs from overlapping. This is useful for multi-level 
// operations that all use WSs: example long range interaction. FFTs at lowest level with a WS acting as an accumulator
template<typename T>
 Array<T>* getWSArray(int nx, int ny, int nz, long level)
{
	T* d;
	T* h;
	getWSMemD(&d, sizeof(T)*nx*ny*nz, level);
	getWSMemH(&h, sizeof(T)*nx*ny*nz, level);
	//printf("New WS Array %p %p\n", d, h);
	Array<T>* a =  new Array<T>(nx,ny,nz,d,h);

	//printf("ddata = %p\n", a->ddata());

	return a;
}

ARRAYCUDA_API dcArray* getWSdcArray(int nx, int ny, int nz, long level)
{
	return getWSArray<doubleComplex>(nx,ny,nz,level);
}
ARRAYCUDA_API fcArray* getWSfcArray(int nx, int ny, int nz, long level)
{
	return getWSArray<floatComplex>(nx,ny,nz,level);
}
ARRAYCUDA_API dArray* getWSdArray(int nx, int ny, int nz, long level)
{
	return getWSArray<double>(nx,ny,nz,level);
}
ARRAYCUDA_API fArray* getWSfArray(int nx, int ny, int nz, long level)
{
	return getWSArray<float>(nx,ny,nz,level);
}
ARRAYCUDA_API iArray* getWSiArray(int nx, int ny, int nz, long level)
{
	return getWSArray<int>(nx,ny,nz,level);
}


template <typename T>
static void matminor(const T* A, const int nx, const int ny, int i, int j, T* dest)
{
	for(int y=0; (y<j) && (y<ny); y++)
	{
		for(int x=0; (x<i) && (x<nx); x++)
		{
			dest[x + (y*(nx-1))] = A[x+nx*y];
		}
		for(int x=i+1; x<nx; x++)
		{
			// printf("A[%i,%i] -> m[%i,%i]\n", x-1,y,x,y);
			dest[(x-1) + (y*(nx-1))] = A[x+nx*y];
		}
	}

	for(int y=j+1; y<ny; y++)
	{
		for(int x=0; x<i && x<nx; x++)
		{
			// printf("A[%i,%i] -> m[%i,%i]\n", x,y-1,x,y);
			dest[x + ((y-1)*(nx-1))] = A[x+nx*y];
		}
		for(int x=i+1; x<nx; x++)
		{
			// printf("A[%i,%i] -> m[%i,%i]\n", x-1,y-1,x,y);
			dest[(x-1) + ((y-1)*(nx-1))] = A[x+nx*y];
		}
	}
}


template <typename T>
static T matdet(const T* A, const int nx, const int ny, bool& ok)
{
	ok = true;
	if(nx != ny)
	{
		ok = false;
		return A[0];
	}

	if(nx == 0)
	{
		return 0;
	}

	if(nx == 1)
	{
		return A[0];
	}

	if(nx == 2)
	{
		return A[0]*A[3] - A[1]*A[2];
	}

	T* m = new T[(nx-1)*(ny-1)];
	T sum = 0;
	for(int i=0; i<nx; i++)
	{
		matminor(A, nx, ny, i, 0, m);

		if(i & 0x1) //odd, negative
		{
			sum = sum - A[i] * matdet(m, nx-1, ny-1, ok);
		}
		else  //even, positive
		{
			sum = sum + A[i] * matdet(m, nx-1, ny-1, ok);
		}
	}
	delete [] m;

	return sum;
}

template <typename T>
static void mm(
	const T* A, const int ra, const int ca,
	const T* B, const int rb, const int cb,
	      T* C, const int rc, const int cc)
{
	for(int r=0; r<ra; r++)
	{
		for(int c=0; c<cb; c++)
		{
			T sum = 0;
			for(int k=0; k<ca; k++)
			{
				sum += A[r*ca + k] * B[k*cb + c];
			}
			C[r*cc + c] = sum;
		}
	}
}
		


template <typename T>
static bool mattrans(const T* src, const int nx, const int ny, T* dest)
{
	if((src == dest) && (nx != ny))
		return false;

	if(src == dest)
	{
		for(int i=0; i<nx; i++)
		{
			for(int j=0; j<i; j++)
			{
				if(i != j)
				{

					const T t = dest[i + nx*j];
					dest[i + nx*j] = dest[j + nx*i];
					dest[j + nx*i] = t;
				}
			}
		}
	}
	else
	{
		for(int i=0; i<nx; i++)
		{
			for(int j=0; j<ny; j++)
			{
				dest[j + i*ny] = src[i + j*nx];
			}
		}
	}
	return true;
}

template <typename T>
static int l_mattrans(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, A, 1);
	
	Array<T>* B = 0;
	if(luaT_is< Array<T> >(L, 2))
		B = luaT_to< Array<T> >(L, 2);
	if(B)
	{
		if(B->ny != A->ny && B->nx != A->nx)
		{
			return luaL_error(L, "Destination array size mismatch");
		}
	}
	else
	{
		B = new Array<T>(A->ny, A->nx);
	}
	

	mattrans(A->data(), A->nx, A->ny, B->data());
	B->new_host = true;
	luaT_push< Array<T> >(L, B);
	return 1;
}



template <typename T>
static int lT_matmul(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, A, 1);
	LUA_PREAMBLE(Array<T>, B, 2);
	
	if(A->nx != B->ny)
		return luaL_error(L, "Column count of A (nx) does not match row count of B (ny)");
	
	Array<T>* C = 0;
	if(luaT_is<Array<T> >(L, 3))
		C = luaT_to< Array<T> >(L, 3);
	else
		C = new Array<T>(B->nx, A->ny);
	
	if(C->nx != B->nx || C->ny != A->ny)
		return luaL_error(L, "Size mismatch for destination matrix");
	
	mm<T>(A->data(), A->ny, A->nx, B->data(), B->ny, B->nx, C->data(), C->ny, C->nx);
	
	C->new_host = true;
	
	luaT_push< Array<T> >(L, C);
	return 1;
}

static int l_matmul(lua_State* L)
{
	if(luaT_is<dArray>(L, 1))
		return lT_matmul<double>(L);
	if(luaT_is<fArray>(L, 1))
		return lT_matmul<float>(L);
	return luaL_error(L, "Array.matMul is only implemented for single and double precision arrays");
}





template <typename T>
static int lT_matdet(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, A, 1);
	
	if(A->nx != A->ny)
		return luaL_error(L, "Matrix is not square");
	
	bool ok;
	
	T res = matdet<T>(A->data(), A->nx, A->ny, ok);
	
	luaT<T>::push(L, res);
    return luaT<T>::elements();
}

static int l_matdet(lua_State* L)
{
	if(luaT_is<dArray>(L, 1))
		return lT_matdet<double>(L);
	if(luaT_is<fArray>(L, 1))
		return lT_matdet<float>(L);
	if(luaT_is<iArray>(L, 1))
		return lT_matdet<int>(L);
	return luaL_error(L, "Array.matDet is only implemented for single and double precision and integer arrays");
}






template<typename T>
static int l_init( Array<T>* a, lua_State* L)
{
	int c[3] = {1,1,1};

	if(lua_istable(L, 1))
	{
		for(int i=0; i<3; i++)
		{
			lua_pushinteger(L, i+1);
			lua_gettable(L, 1);
			c[i] = lua_tointeger(L, -1);
			lua_pop(L, 1);
		}
	}
	else
	{
		for(int i=0; i<3; i++)
			if(lua_isnumber(L, i+1))
				c[i] = lua_tonumber(L, i+1);
	}

	for(int i=0; i<3; i++)
		if(c[i] < 0) c[i] = 0;
	
	a->setSize(c[0], c[1], c[2]);
	return 0;
}








//special cases for complex datatypes (fft):
template <>
const luaL_Reg* Array<doubleComplex>::luaMethods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	merge_luaL_Reg(m, get_base_methods<doubleComplex>());
	merge_luaL_Reg(m, get_fft_methods<doubleComplex>());
	m[127].name = (char*)1;
	return m;
}
template <>
const luaL_Reg* Array<floatComplex>::luaMethods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	merge_luaL_Reg(m, get_base_methods<floatComplex>());
	merge_luaL_Reg(m, get_fft_methods<floatComplex>());
	m[127].name = (char*)1;
	return m;
}
template <typename T>
const luaL_Reg* Array<T>::luaMethods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	merge_luaL_Reg(m, get_base_methods<T>());
	m[127].name = (char*)1;
	return m;
}


template <typename T>
int Array<T>::luaInit(lua_State* L)
{
	return l_init<T>(this, L);
}

// template <typename T>
// int Array<T>::help(lua_State* L)
// {
// 	return 0;
// }



#ifdef WIN32
 #ifdef ARRAY_EXPORTS
  #define ARRAYCUDA_API __declspec(dllexport)
 #else
  #define ARRAYCUDA_API __declspec(dllimport)
 #endif
#else
 #define ARRAYCUDA_API 
#endif

static int l_getmetatable(lua_State* L)
{
	if(!lua_isstring(L, 1))
		return luaL_error(L, "First argument must be a metatable name");
	luaL_getmetatable(L, lua_tostring(L, 1));
	return 1;
}

extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
        
ARRAYCUDA_API int lib_register(lua_State* L);
ARRAYCUDA_API int lib_deps(lua_State* L);
ARRAYCUDA_API int lib_version(lua_State* L);
ARRAYCUDA_API const char* lib_name(lua_State* L);
ARRAYCUDA_API int lib_main(lua_State* L);
}

#include "info.h"
#include "array_luafuncs.h"
ARRAYCUDA_API int lib_register(lua_State* L)
{
#ifdef DOUBLE_ARRAY
	luaT_register<dArray>(L);
#endif
#ifdef SINGLE_ARRAY
	luaT_register<fArray>(L);
#endif
	luaT_register<iArray>(L);
#ifdef DOUBLE_ARRAY
	luaT_register<dcArray>(L);
#endif
#ifdef SINGLE_ARRAY
	luaT_register<fcArray>(L);
#endif

	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");
	if(luaL_dostring(L, __array_luafuncs()))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return luaL_error(L, lua_tostring(L, -1));
	}
	
	return 0;
}

ARRAYCUDA_API int lib_version(lua_State* L)
{
	return __revi;
}

ARRAYCUDA_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Array-Cuda";
#else
	return "Array-Cuda-Debug";
#endif
}


