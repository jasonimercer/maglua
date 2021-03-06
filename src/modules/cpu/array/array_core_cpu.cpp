#include "array_core_cpu.h"
#include "array_core_matrix_cpu.h"
#include "array_core_matrix_cpu_lapack.h"
#include "array.h"


#ifdef ARRAY_CORE_MATRIX_CPU
template<typename T> inline int Array_help_matrix(lua_State* L){return 0;}  
template<>int Array_help_matrix<int>(lua_State* L){return Array_help_matrix_int(L);}
template<>int Array_help_matrix<float>(lua_State* L){return Array_help_matrix_float(L);}
template<>int Array_help_matrix<double>(lua_State* L){return Array_help_matrix_double(L);}
template<>int Array_help_matrix<floatComplex>(lua_State* L){return Array_help_matrix_floatComplex(L);}
template<>int Array_help_matrix<doubleComplex>(lua_State* L){return Array_help_matrix_doubleComplex(L);}


template<typename T> const luaL_Reg* get_base_methods_matrix() {return 0;}
template<> const luaL_Reg* get_base_methods_matrix<int>() {return get_base_methods_matrix_int();}
template<> const luaL_Reg* get_base_methods_matrix<float>() {return get_base_methods_matrix_float();}
template<> const luaL_Reg* get_base_methods_matrix<double>() {return get_base_methods_matrix_double();}
template<> const luaL_Reg* get_base_methods_matrix<floatComplex>() {return get_base_methods_matrix_floatComplex();}
template<> const luaL_Reg* get_base_methods_matrix<doubleComplex>() {return get_base_methods_matrix_doubleComplex();}
#endif



#include <iostream>
#include <vector>
using namespace std;

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
	
	Array<T>* c = 0;
	if(luaT_is< Array<T> >(L, 3))
		c = luaT_to< Array<T> >(L, 3);
	else
		c = new Array<T>(a->nx, a->ny, a->nz);
	
	luaT_push< Array<T> >(L, c);
	
	
	Array<T>::pairwiseMult(c, a, b);
	return 1;
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
	
	if(!Array<T>::pairwiseScaleAdd(c, 1, a, scale, b))
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
static int l_abs(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	Array<T>* b = 0;
	if(luaT_is< Array<T> >(L, 2))
		b = luaT_to< Array<T> >(L, 2);
	else
		b = new Array<T>(a->nx, a->ny, a->nz);

	Array<T>::norm(b, a);

	luaT_push< Array<T> >(L, b);
	return 1;
}

template<typename T>
static int l_pow(lua_State* L)
{
    LUA_PREAMBLE(Array<T>, a, 1);
    double power = lua_tonumber(L, 2);
    Array<T>* b = 0;
    if(luaT_is< Array<T> >(L, 3))
	b = luaT_to< Array<T> >(L, 3);
    else
	b = new Array<T>(a->nx, a->ny, a->nz);
    
    Array<T>::pow(b, a, power);
    
    luaT_push< Array<T> >(L, b);
    return 1;
}

template<typename T>
static int l_resize(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	int n[3] = {1,1,1};
	for(int i=2; i<=4; i++)
	{
		if(lua_isnumber(L, i))
			n[i-1] = lua_tointeger(L, i);
		if(n[i-1] < 1)
			n[i-1] = 1;
	}

	a->setSize(n[0], n[1], n[2]);

	return 0;
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
	double p = 1;
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

template<typename T>
static int l_chop(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	T t = luaT<T>::to(L, 2);
	a->chop(t);
	return 0;
}





template<typename T>
static int l_tally_(lua_State* L)
{
    LUA_PREAMBLE(Array<T>, a, 1);
    
    const int table_pos = 2;
    const int min_pos = 3;
    const int max_pos = 4;

    vector<double> divs;
    vector<int> count;

    double min_val = lua_tonumber(L, min_pos)-1;
    double max_val = lua_tonumber(L, max_pos)+1;

    divs.push_back(min_val);

    if(!lua_istable(L, table_pos))
        return luaL_error(L, "Table expected");

    bool loop = true;
    for(int i=1; loop; i++)
    {
        lua_pushinteger(L, i);
        lua_gettable(L, table_pos);

        if(lua_isnumber(L, -1))
        {
            divs.push_back(lua_tonumber(L, -1));
        }
        else
        {
            loop = false;
        }
        lua_pop(L, 1);
    }
    divs.push_back(max_val);
    
    for(int i=0; i<divs.size()-1; i++)
    {
        count.push_back(0);
    }

    const T* data = a->data();
    for(int i=0; i<a->nxyz; i++)
    { 
        const T v = data[i];
        for(int j=0; j<count.size(); j++)
        {
            const double low  = divs[j];
            const double high = divs[j+1];
            if((low < v) && (v <= high))
            {
                count[j]++;
            }
        }
    }

    lua_newtable(L);
    for(int i=0; i<count.size(); i++)
    {
        lua_pushinteger(L, i+1);
        lua_pushinteger(L, count[i]);
        lua_settable(L, -3);
    }

    return 1;
}

template<typename T> int l_tally(lua_State* L) {return luaL_error(L, "Unimplemented");}
template<> int l_tally<int>(lua_State* L) {return l_tally_<int>(L);}
template<> int l_tally<float>(lua_State* L) {return l_tally_<float>(L);}
template<> int l_tally<double>(lua_State* L) {return l_tally_<double>(L);}





template<typename T>
static int l_stddev_(lua_State* L)
{
    LUA_PREAMBLE(Array<T>, a, 1);
    
    double mean = 0;
    T* d = a->data();
    for(int i=0; i<a->nxyz; i++)
    {
        mean += d[i];
    }
    mean /= (double)(a->nxyz);

    double stddev = 0;
    for(int i=0; i<a->nxyz; i++)
    {
        stddev += pow(d[i]-mean, 2);
    }
    stddev /= (double)(a->nxyz);

    stddev = sqrt(stddev);
    
    lua_pushnumber(L, stddev);

    return 1;
}

template<typename T> int l_stddev(lua_State* L) {return luaL_error(L, "Unimplemented");}
template<> int l_stddev<int>(lua_State* L) {return l_stddev_<int>(L);}
template<> int l_stddev<float>(lua_State* L) {return l_stddev_<float>(L);}
template<> int l_stddev<double>(lua_State* L) {return l_stddev_<double>(L);}

template<typename T>
static int l_totable(lua_State* L)
{
	return luaL_error(L, "not implemented for complex datatypes");
}


template<typename T>
static int l_totable_(lua_State* L)
{
	LUA_PREAMBLE(Array<T>, a, 1);
	int d = 3;
	int r = 0; //num returns
	if(lua_isnumber(L, 2))
	{
		d = lua_tonumber(L, 2);
		if(d < 0) d = 0;
		if(d > 3) d = 3;
	}
	
	if(d == 3)
	{
		lua_newtable(L);
		for(int z=0; z<a->nz; z++)
		{
			lua_pushinteger(L, z+1);
			lua_newtable(L);
			for(int y=0; y<a->ny; y++)
			{
				lua_pushinteger(L, y+1);
				lua_newtable(L);
				for(int x=0; x<a->nx; x++)
				{
					lua_pushinteger(L, x+1);
					const int idx = a->xyz2idx(x,y,z);
					luaT<T>::push(L, a->data()[idx]);
					lua_settable(L, -3);
				}
				lua_settable(L, -3);
			}
			lua_settable(L, -3);
		}
		return 1;
	}
	
	if(d == 2)
	{
		for(int z=0; z<a->nz; z++)
		{
			lua_newtable(L);
			for(int y=0; y<a->ny; y++)
			{
				lua_pushinteger(L, y+1);
				lua_newtable(L);
				for(int x=0; x<a->nx; x++)
				{
					lua_pushinteger(L, x+1);
					const int idx = a->xyz2idx(x,y,z);
					luaT<T>::push(L, a->data()[idx]);
					lua_settable(L, -3);
				}
				lua_settable(L, -3);
			}
		}
		return a->nz;
	}
	if(d == 1)
	{
		for(int z=0; z<a->nz; z++)
		{
			for(int y=0; y<a->ny; y++)
			{
				lua_newtable(L);
				for(int x=0; x<a->nx; x++)
				{
					lua_pushinteger(L, x+1);
					const int idx = a->xyz2idx(x,y,z);
					luaT<T>::push(L, a->data()[idx]);
					lua_settable(L, -3);
				}
			}
		}
		return a->nz*a->ny;
	}
	
	//if(d == 0)
	{
		for(int z=0; z<a->nz; z++)
		{
			for(int y=0; y<a->ny; y++)
			{
				for(int x=0; x<a->nx; x++)
				{
					const int idx = a->xyz2idx(x,y,z);
					luaT<T>::push(L, a->data()[idx]);
				}
			}
		}
		return a->nz*a->ny*a->nx;
	}	
}

template<> int l_totable<float>(lua_State* L) {return l_totable_<float>(L);}
template<> int l_totable<int>(lua_State* L)   {return l_totable_<int>(L);}
template<> int l_totable<double>(lua_State* L){return l_totable_<double>(L);}



static int sort_region(int* r6)
{
	for(int i=0; i<3; i++)
	{
		if(r6[i] > r6[i+3])
		{
			int t = r6[i];
			r6[i] = r6[i+3];
			r6[i+3] = t;
		}
	}
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
// static void get_region(lua_State* L, int idx, int* r6)

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
		{"sameSize",l_sameSize< T >},
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
		{"sum",     l_sum<T>},
		{"absoluted", l_abs<T>},
		{"elementsRaisedTo", l_pow<T>},
		{"resize", l_resize<T>},
		{"scale",   l_scale<T>},
		{"toTable", l_totable<T>},
		{"chop",    l_chop<T>},
		{"_tally",    l_tally<T>},
		{"standardDeviation",    l_stddev<T>},

		{"slice",    l_manip_region<T>},

		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}


template<typename T>
static int l_fft_setup(lua_State* L, Array<T>** a, Array<T>** b)
{
	if(!luaT_is<Array<T> >(L, 1))
	{
		luaL_error(L, "Array expected");
		return 1;
	}
	*a = luaT_to<Array<T> >(L, 1);
	
	*b = 0;
	if(luaT_is<Array<T> >(L, 2))
		*b = luaT_to<Array<T> >(L, 2);
	
	if(*b == 0)
		*b = new Array<T>((*a)->nx, (*a)->ny, (*a)->nz );
	return 0;
}


template<typename T>
static int l_fft1D(lua_State *L)
{
	Array<T>* a;
	Array<T>* b;
	if(l_fft_setup(L, &a, &b)) return 0;
	a->fft1DTo(b, (T*)0);
	luaT_push<Array<T> >(L, b);
	return 1;
}
template<typename T>
static int l_fft2D(lua_State *L)
{
	Array<T>* a;
	Array<T>* b;
	if(l_fft_setup(L, &a, &b)) return 0;
	a->fft2DTo(b, (T*)0);
	luaT_push<Array<T> >(L, b);
	return 1;
}
template<typename T>
static int l_fft3D(lua_State *L)
{
	Array<T>* a;
	Array<T>* b;
	if(l_fft_setup(L, &a, &b)) return 0;
	a->fft3DTo(b, (T*)0);
	luaT_push<Array<T> >(L, b);
	return 1;
}


template<typename T>
static int l_ifft1D(lua_State *L)
{
	Array<T>* a;
	Array<T>* b;
	if(l_fft_setup(L, &a, &b)) return 0;
	a->ifft1DTo(b, (T*)0);
	luaT_push<Array<T> >(L, b);
	return 1;
}
template<typename T>
static int l_ifft2D(lua_State *L)
{
	Array<T>* a;
	Array<T>* b;
	if(l_fft_setup(L, &a, &b)) return 0;
	a->ifft2DTo(b, (T*)0);
	luaT_push<Array<T> >(L, b);
	return 1;
}
template<typename T>
static int l_ifft3D(lua_State *L)
{
	Array<T>* a;
	Array<T>* b;
	if(l_fft_setup(L, &a, &b)) return 0;
	a->ifft3DTo(b, (T*)0);
	luaT_push<Array<T> >(L, b);
	return 1;
}


template<typename C, typename R, int q>
int l_toreal__(lua_State* L)
{
	Array<C>* a = luaT_to< Array<C> >(L, 1);
	Array<R>* b = 0;

	if(luaT_is< Array<R> >(L, 2))
		b = luaT_to< Array<R> >(L, 2);
	else
		b = new Array<R>(a->nx, a->ny, a->nz);

	int nxyz = a->nx * a->ny * a->nz;

	R* r = b->ddata();
	C* c = a->ddata();

	if(q == 0) // real
	{
		for(int i=0; i<nxyz; i++)
			r[i] = c[i].real();
	}
	
	if(q == 1) // imag
	{
		for(int i=0; i<nxyz; i++)
			r[i] = c[i].imag();
	}
	
	if(q == 2) // norm
	{
		for(int i=0; i<nxyz; i++)
			r[i] = c[i].real()*c[i].real() + c[i].imag()*c[i].imag();
	}
	
	luaT_push<Array<R> >(L, b);
	return 1;
}

template<typename T, int q>
int l_toreal_(lua_State* L) {	return luaL_error(L, "not implemented"); }
template<> int l_toreal_<doubleComplex, 0>(lua_State* L) {return l_toreal__<doubleComplex, double, 0>(L); }
template<> int l_toreal_<doubleComplex, 1>(lua_State* L) {return l_toreal__<doubleComplex, double, 1>(L); }
template<> int l_toreal_<doubleComplex, 2>(lua_State* L) {return l_toreal__<doubleComplex, double, 2>(L); }
template<> int l_toreal_< floatComplex, 0>(lua_State* L) {return l_toreal__< floatComplex,  float, 0>(L); }
template<> int l_toreal_< floatComplex, 1>(lua_State* L) {return l_toreal__< floatComplex,  float, 1>(L); }
template<> int l_toreal_< floatComplex, 2>(lua_State* L) {return l_toreal__< floatComplex,  float, 2>(L); }


template<typename T, int q>
static int l_toreal(lua_State* L)
{
	return l_toreal_<T,q>(L);
}


template<typename F, typename C>
static int l_tocomplex_(lua_State* L)
{
	LUA_PREAMBLE(Array<F>, src, 1);

	Array<C>* b;
	if(luaT_is<Array<C> >(L, 2))
		b = luaT_to<Array<C> >(L, 2);
	else
		b = new Array<C>(src->nx, src->ny, src->nz);
	
	b->zero(); 
	arraySetRealPart(b->ddata(), src->ddata(), src->nx*src->ny*src->nz);
	luaT_push< Array<C> >(L, b);
	return 1;
}

template<typename T>
int l_tocomplex(lua_State* L)
{
	return luaL_error(L, "Not implemented");
}
template<>
int l_tocomplex<float>(lua_State* L)
{
	return l_tocomplex_<float, floatComplex>(L);
}
template<>
int l_tocomplex<double>(lua_State* L)
{
	return l_tocomplex_<double, doubleComplex>(L);
}


template<typename T>
int Array_help_fp(lua_State* L)
{
	lua_CFunction func = lua_tocfunction(L, 1);

	if(func == &(l_tocomplex<T>))
	{
		lua_pushstring(L, "Store array data in the real component of a complex array");
		lua_pushstring(L, "1 Optional Array of same complex type: Destination array, a new array will be created if not supplied");
		lua_pushstring(L, "1 Complex Array: Source array augmented to complex");
		return 3;
	}
	return 0;
}

// this nonsense looks dumb but it gets around some compiler problems
static bool FC1(lua_CFunction lhs, lua_CFunction rhs)
{
	return lhs == rhs;
}
// function compare
#define if_FC1(lhs, rhs) if(FC1(lhs,rhs))
#define if_FC2(lhs, rhs1, rhs2) if(FC1(lhs,rhs1) && FC1(lhs,rhs2))


template<typename T>
int Array_help_fft_complex(lua_State* L)
{
	lua_CFunction func = lua_tocfunction(L, 1);
		
// 	if(func == &(l_fft1D<T>))
	if_FC1(func, l_fft1D<T>)
	{
		lua_pushstring(L, "1D Fourier Transform an array along the X direction");
		lua_pushstring(L, "1 Optional Array of same type: Destination of transform, a new array will be created if not supplied");
		lua_pushstring(L, "1 Array: The result of the transform");
		return 3;
	}	
// 	if(func == &(l_fft2D<T>))
	if_FC1(func, l_fft2D<T>)
	{
		lua_pushstring(L, "2D Fourier Transform an array along the X and Y directions");
		lua_pushstring(L, "1 Optional Array of same type: Destination of transform, a new array will be created if not supplied");
		lua_pushstring(L, "1 Array: The result of the transform");
		return 3;
	}	
// 	if(func == &(l_fft3D<T>))
	if_FC1(func, l_fft3D<T>)
	{
		lua_pushstring(L, "3D Fourier Transform an array along the X, Y and Z directions");
		lua_pushstring(L, "1 Optional Array of same type: Destination of transform, a new array will be created if not supplied");
		lua_pushstring(L, "1 Array: The result of the transform");
		return 3;
	}	
		
// 	if(func == &(l_ifft1D<T>))
	if_FC1(func, l_ifft1D<T>)
	{
		lua_pushstring(L, "1D Inverse Fourier Transform an array along the X direction");
		lua_pushstring(L, "1 Optional Array of same type: Destination of transform, a new array will be created if none is supplied");
		lua_pushstring(L, "1 Array: The result of the transform");
		return 3;
	}	
// 	if(func == &(l_ifft2D<T>))
	if_FC1(func, l_ifft2D<T>)
	{
		lua_pushstring(L, "2D Inverse Fourier Transform an array along the X and Y directions");
		lua_pushstring(L, "1 Optional Array of same type: Destination of transform, a new array will be created if none is supplied");
		lua_pushstring(L, "1 Array: The result of the transform");
		return 3;
	}	
// 	if(func == &(l_ifft3D<T>))
	if_FC1(func, l_ifft3D<T>)
	{
		lua_pushstring(L, "3D Inverse Fourier Transform an array along the X, Y and Z directions");
		lua_pushstring(L, "1 Optional Array of same type: Destination of transform, a new array will be created if none is supplied");
		lua_pushstring(L, "1 Array: The result of the transform");
		return 3;
	}	

// 	if(func == &(l_toreal<T,0>))
	if_FC1(func, (l_toreal<T,0>))
	{
		lua_pushstring(L, "Copy real component of data to a real array");
		lua_pushstring(L, "1 Optional Array of same real type: Destination array, a new array will be created if none is supplied");
		lua_pushstring(L, "1 Array: The data");
		return 3;
	}	
	if_FC1(func, (l_toreal<T,1>))
// 	if(func == &(l_toreal<T,1>))
	{
		lua_pushstring(L, "Copy imaginary component of data to a real array");
		lua_pushstring(L, "1 Optional Array of same real type: Destination array, a new array will be created if none is supplied");
		lua_pushstring(L, "1 Array: The data");
		return 3;
	}	
	if_FC1(func, (l_toreal<T,2>))
// 	if(func == &(l_toreal<T,2>))
	{
		lua_pushstring(L, "Copy norm of data to a real array");
		lua_pushstring(L, "1 Optional Array of same real type: Destination array, a new array will be created if none is supplied");
		lua_pushstring(L, "1 Array: The data");
		return 3;
	}	

	return 0;
}

template<typename T> //specializations for types
int Array_help_specialization(lua_State* L)
{
	return 0;
}

//special cases for complex datatypes 
template <>
int Array_help_specialization<doubleComplex>(lua_State* L)
{
	int r = 0;
	r = Array_help_fft_complex<doubleComplex>(L); if(r) return r;
#ifdef ARRAY_CORE_MATRIX_CPU
	r = Array_help_matrix<doubleComplex>(L); if(r) return r;
#endif
	return 0;

}
template <>
int Array_help_specialization<floatComplex>(lua_State* L)
{
	int r = 0;
	r = Array_help_fft_complex<floatComplex>(L); if(r) return r;
#ifdef ARRAY_CORE_MATRIX_CPU
	r = Array_help_matrix<floatComplex>(L); if(r) return r;
#endif	
	return 0;
}
//special cases for floating point datatypes 
template <>
int Array_help_specialization<double>(lua_State* L)
{
	int r = 0;
	r = Array_help_fp<double>(L); if(r) return r;
#ifdef ARRAY_CORE_MATRIX_CPU
	r = Array_help_matrix<double>(L); if(r) return r;
#endif
	return 0;
}
template <>
int Array_help_specialization<float>(lua_State* L)
{
	int r = 0;
	r = Array_help_fp<float>(L); if(r) return r;
#ifdef ARRAY_CORE_MATRIX_CPU
	r = Array_help_matrix<float>(L); if(r) return r;
#endif
	return 0;
}



template<typename T>
static int Array_help(lua_State* L)
{
	if(lua_gettop(L) == 0)
	{
		lua_pushstring(L, "Class for 3D Data Arrays");
		lua_pushstring(L, "0 to 3 Integers, Optional table: Length of each dimension X, Y and Z. Default values are 1. If a table is found after the size then the values in it will be used to populate the array.");
		lua_pushstring(L, ""); //output, empty
		return 3;
	}

	lua_CFunction func = lua_tocfunction(L, 1);
	
	lua_CFunction f01 = l_sameSize<T>; //THIS IS BULLSHIT! Doing this so that icc 11 can compile cleanly. 
	if(func == f01)
	{
		lua_pushstring(L, "Test if a given array has the same dimensions");
		lua_pushstring(L, "1 Array");
		lua_pushstring(L, "1 Boolean: True if sizes match");
		return 3;
	}
	lua_CFunction f02 = l_setAll<T>;
	if(func == f02)
	{
		lua_pushstring(L, "Set all values in the array to the given value");
		lua_pushstring(L, "1 Value: The new value for all element entries");
		lua_pushstring(L, "");
		return 3;
	}
	lua_CFunction f03 = l_pwm<T>;
	if(func == f03)
	{
		lua_pushstring(L, "Multiply each data in this array with the data in another storing in a destination array");
		lua_pushstring(L, "1 Source Array, 1 Optional Destination array: The pairwise scaling array and the destination array");
		lua_pushstring(L, "1 Destination Array: If a destination array is not supplied it will be created.");
		return 3;
	}
	lua_CFunction f03b = l_pwsa<T>;
	if(func == f03b)
	{
		lua_pushstring(L, "Add each data in this array with a constant factor times the data in another storing in a destination array");
		lua_pushstring(L, "1 Value, 1 Source Array, 1 Optional Destination Array: The factor to scale the source array and the destination of the pairwise addition");
		lua_pushstring(L, "1 Destination Array: If no destination array is supplied, it will be created");
		return 3;
	}
	lua_CFunction f04 = l_dot<T>;
	if(func == f04)
	{
		lua_pushstring(L, "Compute the dot product of the current array and another of equal size and type");
		lua_pushstring(L, "1 Arrays: other array");
		lua_pushstring(L, "1 Value: result of dot product");
		return 3;
	}
	
	lua_CFunction f05 = l_zero<T>;
	if(func == f05)
	{
		lua_pushstring(L, "Set all values in the array 0");
		lua_pushstring(L, "");
		lua_pushstring(L, "");
		return 3;
	}
	lua_CFunction f06 = l_get_nx<T>;
	if(func == f06)
	{
		lua_pushstring(L, "Return the size of the X dimension");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size fo the X dimension");
		return 3;
	}
	lua_CFunction f07 = l_get_ny<T>;
	if(func == f07)
	{
		lua_pushstring(L, "Return the size of the Y dimension");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size fo the Y dimension");
		return 3;
	}
	lua_CFunction f08 = l_get_nz<T>;
	if(func == f08)
	{
		lua_pushstring(L, "Return the size of the Z dimension");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: Size fo the Z dimension");
		return 3;
	}

	lua_CFunction f09 = l_get<T>;
	if(func == f09)
	{
		lua_pushstring(L, "Get an element from the array");
		lua_pushstring(L, "1, 2 or 3 integers (or 1 table): indices(XYZ) of the element to fetch default values are 1");
		lua_pushstring(L, "1 value");
		return 3;
	}
	lua_CFunction f10 = l_set<T>;
	if(func == f10)
	{
		lua_pushstring(L, "Set an element of the array");
		lua_pushstring(L, "1, 2 or 3 integers (or 1 table), 1 value or 1 Array: indices(XYZ) of the element to set, default values are 1. Last argument is the new value. If the last argument is an array then all elements in the array are copied to the calling object.");
		lua_pushstring(L, "");
		return 3;
	}
	lua_CFunction f10b = l_setfromtable<T>;
	if(func == f10b)
	{
		lua_pushstring(L, "Set elements of the array based on table values");
		lua_pushstring(L, "1 1D, 2D or 3D table of values: new values to set");
		lua_pushstring(L, "");
		return 3;
	}
	lua_CFunction f10c = l_copy<T>;
	if(func == f10c)
	{
		lua_pushstring(L, "Create a copy of an array");
		lua_pushstring(L, "1 Optional Array: Destination array, if it is not provided a new array will be created.");
		lua_pushstring(L, "1 Array: A copy of the array");
		return 3;
	}
	lua_CFunction f11 = l_addat<T>;
	if(func == f11)
	{
		lua_pushstring(L, "Add a value to an element of the array");
		lua_pushstring(L, "1, 2 or 3 integers (or 1 table), 1 value: indices(XYZ) of the element to modify, default values are 1. Last argument is the value to add");
		lua_pushstring(L, "");
		return 3;
	}
	lua_CFunction f12 = l_min<T>;
	if(func == f12)
	{
		lua_pushstring(L, "Find minimum value and corresponding index");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 value and 1 integer");
		return 3;
	}
	lua_CFunction f13 = l_max<T>;
	if(func == f13)
	{
		lua_pushstring(L, "Find maximum value and corresponding index");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 value and 1 integer");
		return 3;
	}
	lua_CFunction f14 = l_mean<T>;
	if(func == f14)
	{
		lua_pushstring(L, "Find mean of array");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 value");
		return 3;
	}
	lua_CFunction f15 = l_sum<T>;
	if(func == f15)
	{
		lua_pushstring(L, "Calculate sum of array");
		lua_pushstring(L, "1 optional Number: The power each element will be raised to before summing. Default = 1.");
		lua_pushstring(L, "1 value");
		return 3;
	}
	lua_CFunction f15b = l_abs<T>;
	if(func == f15b)
	{
		lua_pushstring(L, "Take the absolute value of all elements.");
		lua_pushstring(L, "1 optional Array: Destination. If empty, a new array will be created.");
		lua_pushstring(L, "1 Array: The absolute value of the calling array.");
		return 3;
	}
	lua_CFunction f15q = l_pow<T>;
	if(func == f15q)
	{
		lua_pushstring(L, "Raise each element to the given power.");
		lua_pushstring(L, "1 Number, 1 optional Array: Power to raise each element. Destination, if empty a new array will be created.");
		lua_pushstring(L, "1 Array: Array with elements raised to the given power");
		return 3;
	}
	lua_CFunction f15c = l_resize<T>;
	if(func == f15c)
	{
		lua_pushstring(L, "Resize an array. New values may not be related to old values.");
		lua_pushstring(L, "Up to 3 integers: New dimensions.");
		lua_pushstring(L, "");
		return 3;
	}
	lua_CFunction f16 = l_scale<T>;
	if(func == f16)
	{
		lua_pushstring(L, "Scale all values in the array by the given value");
		lua_pushstring(L, "1 Value: The scaling factor.");
		lua_pushstring(L, "");
		return 3;
	}
	lua_CFunction f123 = l_chop<T>;
	if(func == f123)
	{
		lua_pushstring(L, "Make small values equal to zero. Similar to Mathematica's Chop function.");
		lua_pushstring(L, "1 Value: The tolerance, values with smaller norms than this tolerance's norm will be set to zero.");
		lua_pushstring(L, "");
		return 3;
	}
	lua_CFunction f123x = l_stddev<T>;
	if(func == f123x)
	{
            lua_pushstring(L, "Compute standard deviation");
            lua_pushstring(L, "");
            lua_pushstring(L, "1 Number: Standard deviation");
            return 3;
	}
	lua_CFunction f16b = l_totable<T>;
	if(func == f16b)
	{
		lua_pushstring(L, "Convert the array into a Lua table");
		lua_pushstring(L, "1 Optional Integer: Number of dimensions. By default it will be a 3D table, this may not be the desired dimensionality. If fewer dimensions are requested than the dimensionality of the array then multiple return values will be given.");
		lua_pushstring(L, "1 or more Tables or 1 or more Numbers: If the dimensionality requested is 0 then numbers will be returned otherwise tables will be returned.");
		return 3;
	}

	lua_CFunction f17 = l_manip_region<T>;
	if(func == f17)
	{
		lua_pushstring(L, "Extract a region of an array or copy to another region of another array (or same array)");
		lua_pushstring(L, "1 Table of Tables, 1 Optional Array, 1 Optional Table of Tables: The first argument represents the min and max corners of the slice that will be copied from the calling Array, the coordinates can be up to 3 integers long representing Cartesian coordinates, missing dimensions will be assumed to be 1. If an Array is given then the slice will be copied into it at the provided optional location, if an Array is not supplied then a new one will be created. The last optional table is the destination region for the data in the destination table. The size of the source and destination slices must match.");
		lua_pushstring(L, "1 Array: Either the supplied Array with the data copied in or a new array.");
		return 3;
	}


	int r1;
#ifdef ARRAY_CORE_MATRIX_CPU_LAPACK
	r1 = l_mat_lapack_help(L);
	if(r1) return r1;
#endif
	
	int r = Array_help_specialization<T>(L);
	if(r) return r;

	return LuaBaseObject::help(L);
}


// These are called from a macro in the header
int array_help_specialization_int(lua_State* L)
{
	return Array_help<int>(L);
}
int array_help_specialization_float(lua_State* L)
{
	return Array_help<float>(L);
}
int array_help_specialization_double(lua_State* L)
{
	return Array_help<double>(L);
}
int array_help_specialization_floatComplex(lua_State* L)
{
	return Array_help<floatComplex>(L);
}
int array_help_specialization_doubleComplex(lua_State* L)
{
	return Array_help<doubleComplex>(L);
}


template<typename T>
static const luaL_Reg* get_real_methods()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	static const luaL_Reg _m[] =
	{
		{"toComplex",   l_tocomplex<T>},
		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}



template<typename T>
static const luaL_Reg* get_complex_methods()
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

		{"toRealR", l_toreal<T,0>},
		{"toRealI", l_toreal<T,1>},
		{"toRealN", l_toreal<T,2>},

		{NULL, NULL}
	};
	merge_luaL_Reg(m, _m);
	m[127].name = (char*)1;
	return m;
}











template<typename T>
static int l_init( Array<T>* a, lua_State* L)
{
	int c[3] = {1,1,1};
        int data_start = 1;

        for(int i=0; i<3; i++)
            if(lua_isnumber(L, i+1))
            {
                c[i] = lua_tonumber(L, i+1);
                data_start = i+2;
            }

	for(int i=0; i<3; i++)
            if(c[i] < 0) 
                c[i] = 0;
	
	a->setSize(c[0], c[1], c[2]);

	// looking for a table with initial values
	int n = c[0]  * c[1] * c[2];
        if(lua_istable(L, data_start))
        {
            int idx = 0;
            for(int i=0; i<n; i++)
            {
                lua_pushinteger(L, i+1);
                lua_gettable(L, data_start);
                a->data()[i] = luaT<T>::to(L, -1);
                lua_pop(L, 1);
            }
        }

	return 0;
}





template <typename T>
luaL_Reg* Array_luaMethods()
{
	static luaL_Reg mm[128] = {_NULLPAIR128};
	if(mm[127].name)	return mm;
	
	merge_luaL_Reg(mm, get_base_methods<T>());
	mm[127].name = (char*)1;
	return mm;
}

//special cases for floating point datatypes
template <>
luaL_Reg* Array_luaMethods<double>()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	merge_luaL_Reg(m, get_base_methods<double>());
	merge_luaL_Reg(m, get_real_methods<double>());
#ifdef ARRAY_CORE_MATRIX_CPU
	merge_luaL_Reg(m, get_base_methods_matrix<double>());	
#endif
	m[127].name = (char*)1;
	return m;
}

template <>
luaL_Reg* Array_luaMethods<float>()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	merge_luaL_Reg(m, get_base_methods<float>());
	merge_luaL_Reg(m, get_real_methods<float>());
#ifdef ARRAY_CORE_MATRIX_CPU
	merge_luaL_Reg(m, get_base_methods_matrix<float>());	
#endif
	m[127].name = (char*)1;
	return m;
}


//special cases for complex datatypes (fft):
template <>
luaL_Reg* Array_luaMethods<doubleComplex>()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	merge_luaL_Reg(m, get_base_methods<doubleComplex>());
	merge_luaL_Reg(m, get_complex_methods<doubleComplex>());
	m[127].name = (char*)1;
	return m;
}

template <>
luaL_Reg* Array_luaMethods<floatComplex>()
{
	static luaL_Reg m[128] = {_NULLPAIR128};
	if(m[127].name)	return m;
	merge_luaL_Reg(m, get_base_methods<floatComplex>());
	merge_luaL_Reg(m, get_complex_methods<floatComplex>());
	m[127].name = (char*)1;
	return m;
}

// these are called from the .h file
const luaL_Reg* array_luamethods_specialization_int()
{
  return Array_luaMethods<int>();
}
const luaL_Reg* array_luamethods_specialization_float()
{
  return Array_luaMethods<float>();
}
const luaL_Reg* array_luamethods_specialization_double()
{
  return Array_luaMethods<double>();
}

const luaL_Reg* array_luamethods_specialization_doubleComplex()
{
  return Array_luaMethods<doubleComplex>();
}
const luaL_Reg* array_luamethods_specialization_floatComplex()
{
  return Array_luaMethods<floatComplex>();
}

int array_luainit_specialization_int(Array<int>* that, lua_State* L)
{
  return l_init<int>(that, L);
}
int array_luainit_specialization_float(Array<float>* that, lua_State* L)
{
  return l_init<float>(that, L);
}
int array_luainit_specialization_double(Array<double>* that, lua_State* L)
{
  return l_init<double>(that, L);
}
int array_luainit_specialization_floatComplex(Array<floatComplex>* that, lua_State* L)
{
  return l_init<floatComplex>(that, L);
}
int array_luainit_specialization_doubleComplex(Array<doubleComplex>* that, lua_State* L)
{
  return l_init<doubleComplex>(that, L);
}

/*
template <typename T>
int Array<T>::luaInit(lua_State* L)
{
	return l_init<T>(this, L);
}
*/

// template <typename T>
// int Array<T>::help(lua_State* L)
// {
// 	return 0;
// }






// the level argument below prevents WSs from overlapping. This is useful for multi-level 
// operations that all use WSs: example long range interaction. FFTs at lowest level with a WS acting as an accumulator
template<typename T>
 Array<T>* getWSArray(int nx, int ny, int nz, long level)
{
	T* m;
	T* h;
	getWSMem(&m, sizeof(T)*nx*ny*nz, level);
	//printf("New WS Array %p %p\n", d, h);
	Array<T>* a =  new Array<T>(nx,ny,nz,m);

	//printf("ddata = %p\n", a->ddata());

	return a;
}

ARRAY_API dcArray* getWSdcArray(int nx, int ny, int nz, long level)
{
	return getWSArray<doubleComplex>(nx,ny,nz,level);
}
ARRAY_API fcArray* getWSfcArray(int nx, int ny, int nz, long level)
{
	return getWSArray<floatComplex>(nx,ny,nz,level);
}
ARRAY_API dArray* getWSdArray(int nx, int ny, int nz, long level)
{
	return getWSArray<double>(nx,ny,nz,level);
}
ARRAY_API fArray* getWSfArray(int nx, int ny, int nz, long level)
{
	return getWSArray<float>(nx,ny,nz,level);
}
ARRAY_API iArray* getWSiArray(int nx, int ny, int nz, long level)
{
	return getWSArray<int>(nx,ny,nz,level);
}











#ifdef WIN32
 #ifdef ARRAY_EXPORTS
  #define ARRAY_API __declspec(dllexport)
 #else
  #define ARRAY_API __declspec(dllimport)
 #endif
#else
 #define ARRAY_API 
#endif


static int l_getmetatable(lua_State* L)
{
	if(!lua_isstring(L, 1))
		return luaL_error(L, "First argument must be a metatable name");
	luaL_getmetatable(L, lua_tostring(L, 1));
	return 1;
}


static int l_get_dc_ws(lua_State* L)
{
	int nx = lua_tointeger(L, 1);
	int ny = lua_tointeger(L, 2);
	int nz = lua_tointeger(L, 3);
	const char* name = lua_tostring(L, 4);

	if(nx*ny*nz == 0)
		return luaL_error(L, "Must supply non-zero nx,ny and nz");

	if(name == 0)
		return luaL_error(L, "Must supply name");

	luaT_push<dcArray>(L,  getWSdcArray(nx,ny,nz,hash32(name)));
	return 1;
}


static int l_get_d_ws(lua_State* L)
{
    int nx = lua_tointeger(L, 1);
    int ny = lua_tointeger(L, 2);
    int nz = lua_tointeger(L, 3);
    const char* name = lua_tostring(L, 4);

    if(nx*ny*nz == 0)
        return luaL_error(L, "Must supply non-zero nx,ny and nz");

    if(name == 0)
        return luaL_error(L, "Must supply name");

    luaT_push<dArray>(L,  getWSdArray(nx,ny,nz,hash32(name)));
    return 1;
}

static int l_registerws(lua_State* L)
{
	registerWS();
	return 0;
}
static int l_unregisterws(lua_State* L)
{
	unregisterWS();
	return 0;
}


extern "C"
{
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
        
ARRAY_API int lib_register(lua_State* L);
ARRAY_API int lib_deps(lua_State* L);
ARRAY_API int lib_version(lua_State* L);
ARRAY_API const char* lib_name(lua_State* L);
ARRAY_API int lib_main(lua_State* L);
}


#include "array_luafuncs.h"
#include "info.h"
ARRAY_API int lib_register(lua_State* L)
{
//  	dArray foo1;

#ifdef DOUBLE_ARRAY
	luaT_register< Array<double> >(L);
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

  	lua_getglobal(L, "Array");
	lua_pushstring(L, "DoubleComplexWorkSpace");
	lua_pushcfunction(L, l_get_dc_ws);
	lua_settable(L, -3);
	lua_pushstring(L, "DoubleWorkSpace");
	lua_pushcfunction(L, l_get_d_ws);
	lua_settable(L, -3);

	lua_pushstring(L, "_registerws");
	lua_pushcfunction(L, l_registerws);
	lua_settable(L, -3);
	lua_pushstring(L, "_unregisterws");
	lua_pushcfunction(L, l_unregisterws);
	lua_settable(L, -3);

	lua_pushstring(L, "WorkSpaceInfo");
	lua_pushcfunction(L, l_ws_info);
	lua_settable(L, -3);

	lua_pop(L, 1);

	
    lua_pushcfunction(L, l_getmetatable);
    lua_setglobal(L, "maglua_getmetatable");

    luaL_dofile_array_luafuncs(L);

    lua_pushnil(L);
    lua_setglobal(L, "maglua_getmetatable");

	return 0;
}

ARRAY_API int lib_version(lua_State* L)
{
	return __revi;
}

ARRAY_API const char* lib_name(lua_State* L)
{
#if defined NDEBUG || defined __OPTIMIZE__
	return "Array";
#else
	return "Array-Debug";
#endif
}


ARRAY_API int lib_main(lua_State* L)
{
	return 0;

}

