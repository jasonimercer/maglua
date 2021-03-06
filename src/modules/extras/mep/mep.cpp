#include "mep.h"
#include "mep_luafuncs.h"
#include "info.h"
#include "luamigrate.h"
#include <math.h>

#include <algorithm>
#include <numeric>

#include "ddd.h"

// deterministic random number generator - don't want to interfere with
// the internal state of the C generator. Too lazy to use stateful rng
static unsigned long next = 1;


/* RAND_MAX assumed to be 32767 */
static int myrand(void) {
    next = next * 1103515245 + 12345;
    return((unsigned)(next/65536) % 32768);
}
static double myrandf()
{
    return (double)(myrand()) / 32767.0;
}


#define _acos(x) _acosFL(x, __FILE__, __LINE__)

static double _acosFL(double x, const char* file, const int line)
{
    if(fabs(x) > 1)
    {
	{int* i = (int*)5; *i = 5;}
	fprintf(stderr, "(%s:%i) argument of acos out of range (%g)\n", file, line, x);
	if(x < 0)
	    x = -1;
	else
	    x = 1;
    }
    return acos(x);
}


#define CARTESIAN_X 0
#define CARTESIAN_Y 1
#define CARTESIAN_Z 2

#define SPHERICAL_R 0
#define SPHERICAL_PHI 1
#define SPHERICAL_THETA 2

#define CANONICAL_R 0
#define CANONICAL_PHI 1
#define CANONICAL_P 2



static double dot(const double* a, const double* b, int n = 3)
{
    double s = 0;
    for(int i=0; i<n; i++)
	s += a[i]*b[i];
    return s;
}


MEP::MEP()
    : LuaBaseObject(hash32(MEP::slineage(0)))
{
    ref_data = LUA_REFNIL;
    ref_energy_function = LUA_REFNIL;
    fitEnergy = 0;
    ss = 0;
    energy_ok = false;
    good_distances = false;
    beta = 0.1;
    relax_direction_fail_max = 4;
    epsilon = 1e-3;
    fixedRadius = true;
}

MEP::~MEP()
{
    deinit();
	
}


static double cart_dot(vector<VectorCS>& vec, vector<VectorCS>& vec2)
{
    double s = 0;
    for(int i=0; i<vec.size(); i++)
    {
	for(int j=0; j<3; j++)
	{
	    s += vec[i].v[j] * vec2[i].v[j];
	}
    }
    return s;
}

static double cart_norm2(vector<VectorCS>& vec)
{
    return cart_dot(vec, vec);
}

static double cart_norm2(double* vec, int n)
{
    double s = 0;
    for(int i=0; i<n; i++)
    {
	s += vec[i] * vec[i];
    }
    return s;
}

static double cart_norm(double* vec, int n)
{
    return sqrt(cart_norm2(vec, n));
}


void MEP::init()
{
}

void MEP::deinit()
{
    if(ref_data != LUA_REFNIL)
	luaL_unref(L, LUA_REGISTRYINDEX, ref_data);
    if(ref_energy_function != LUA_REFNIL)
	luaL_unref(L, LUA_REGISTRYINDEX, ref_energy_function);

    luaT_dec<FitEnergy>(fitEnergy);
    fitEnergy = 0;

    ref_data = LUA_REFNIL;
    ref_energy_function = LUA_REFNIL;

    state_path.clear();
    image_site_mobility.clear();
    sites.clear();

    ss = luaT_dec<SpinSystem>(ss);
}

int MEP::lua_setspinsystem(lua_State* L)
{
    luaT_set<SpinSystem>(&ss, luaT_to<SpinSystem>(L, 2));
    return 0;
}

int MEP::lua_getspinsystem(lua_State* L)
{
    luaT_push<SpinSystem>(L, ss);
    return 1;
}

int MEP::lua_setEnergyFunction(lua_State* L)
{
    if(ref_energy_function != LUA_REFNIL)
    {
        luaL_unref(L, LUA_REGISTRYINDEX, ref_energy_function);
        ref_energy_function = LUA_REFNIL;
    }

    if(luaT_is<FitEnergy>(L, 2))
    {
        FitEnergy* fe = luaT_to<FitEnergy>(L, 2);
        luaT_set<FitEnergy>(&fitEnergy, fe);
    }
    else
    {
        if(lua_type(L, 2) == LUA_TFUNCTION)
        {
            ref_energy_function = luaL_ref(L, LUA_REGISTRYINDEX);
        }
    }

    energy_ok = false;
    return 0;
}


int MEP::lua_getEnergyFunction(lua_State* L)
{
    if(fitEnergy)
    {
        luaT_push<FitEnergy>(L, fitEnergy);
        return 1;
    }

    lua_rawgeti(L, LUA_REGISTRYINDEX, ref_energy_function);
    return 1;
}



int MEP::luaInit(lua_State* L, int base)
{
    state_path.clear();
    image_site_mobility.clear();
    sites.clear();
	
    return LuaBaseObject::luaInit(L, base);
}

void MEP::encode(buffer* b) //encode to data stream
{
    ENCODE_PREAMBLE
	if(ref_data != LUA_REFNIL)
	{
	    lua_rawgeti(L, LUA_REGISTRYINDEX, ref_data);
	}
	else
	{
	    lua_pushnil(L);
	}

    _exportLuaVariable(L, -1, b);
    lua_pop(L, 1);


    if(ref_energy_function != LUA_REFNIL)
    {
        lua_rawgeti(L, LUA_REGISTRYINDEX, ref_energy_function);
    }
    else
    {
        lua_pushnil(L);
    }

    // TODo: ENCODE FITENERGY

    _exportLuaVariable(L, -1, b);
    lua_pop(L, 1);


	
    encodeInteger(fixedRadius, b);
    encodeInteger(state_path.size(), b);
    for(int i=0; i<state_path.size(); i++)
    {
	encodeDouble(state_path[i].v[0], b);
	encodeDouble(state_path[i].v[1], b);
	encodeDouble(state_path[i].v[2], b);
	encodeInteger((int)state_path[i].cs, b);
    }
	
    encodeInteger(image_site_mobility.size(), b);
    for(int i=0; i<image_site_mobility.size(); i++)
    {
	encodeDouble(image_site_mobility[i], b);
    }
	
    encodeInteger(sites.size(), b);
    for(int i=0; i<sites.size(); i++)
    {
	encodeInteger(sites[i], b);
    }

    encodeDouble(beta, b);
    encodeDouble(epsilon, b);
    encodeInteger(relax_direction_fail_max, b);
}

int  MEP::decode(buffer* b) // decode from data stream
{
    deinit();
    _importLuaVariable(L, b);
    ref_data = luaL_ref(L, LUA_REGISTRYINDEX);
    _importLuaVariable(L, b);
    ref_energy_function = luaL_ref(L, LUA_REGISTRYINDEX);
    
    // TODO: DECODE FITENERGY

    fixedRadius = decodeInteger(b);
	
    int n = decodeInteger(b);
    for(int i=0; i<n; i++)
    {
	double v0 = decodeDouble(b);
	double v1 = decodeDouble(b);
	double v2 = decodeDouble(b);
	CoordinateSystem cs = (CoordinateSystem)decodeInteger(b);
	state_path.push_back(VectorCS(v0,v1,v2,cs));
    }

    n = decodeInteger(b);
    for(int i=0; i<n; i++)
    {
	image_site_mobility.push_back(decodeDouble(b));
    }
    n = decodeInteger(b);
    for(int i=0; i<n; i++)
    {
	sites.push_back(decodeInteger(b));
    }

    beta = decodeDouble(b);
    epsilon = decodeDouble(b);
    relax_direction_fail_max = decodeInteger(b);

    return 0;
}


void MEP::get_site_ss(int px, int py, int pz, double* v3)
{
    const int c = ss->getidx(px, py, pz);

    v3[0] = ss->x->data()[c];
    v3[1] = ss->y->data()[c];
    v3[2] = ss->z->data()[c];
}

bool MEP::set_site_ss(int px, int py, int pz, double* v3)
{
    double current[4];
    const int c = ss->getidx(px, py, pz);

    ss->get(c, current);

    double m = ss->ms->data()[c];
    ss->set(px,py,pz, v3[0], v3[1], v3[2], m);

    return memcpy(current, v3, sizeof(double)*3) != 0;
}

double MEP::get_energy_ss(lua_State* L)
{
    lua_rawgeti(L, LUA_REGISTRYINDEX, ref_energy_function);
    luaT_push<SpinSystem>(L, ss);

    lua_call(L, 1, 1);
    double e = lua_tonumber(L, -1);
    lua_pop(L, 1);

    return e;
}


void MEP::internal_copy_to(MEP* dest)
{
    dest->state_path.clear();
    dest->fixedRadius = fixedRadius;

    lua_rawgeti(L, LUA_REGISTRYINDEX, ref_energy_function);
    dest->ref_energy_function = luaL_ref(L, LUA_REGISTRYINDEX);

    luaT_set<FitEnergy>(&(dest->fitEnergy), fitEnergy);

    for(int i=0; i<state_path.size(); i++)
    {
	dest->state_path.push_back(state_path[i].copy());
    }
	
    dest->image_site_mobility.clear();
    for(int i=0; i<image_site_mobility.size(); i++)
    {
	dest->image_site_mobility.push_back(image_site_mobility[i]);
    }

    dest->sites.clear();
    for(unsigned int i=0; i<sites.size(); i++)
    {
	dest->sites.push_back(sites[i]);
    }

    luaT_set<SpinSystem>(&(dest->ss), ss);

    dest->path_tangent.clear();
    dest->force_vector.clear();
    dest->energies.clear();

    //dest->state_xyz_path = state_xyz_path;
    //dest->sites = sites;
    //dest->path_tangent = path_tangent;
    //dest->force_vector = force_vector;
    //dest->energies = energies;
    dest->energy_ok = false;
    dest->beta = beta;
}


// return a list of what the user can play with
// not telling them about "Undefined"
int MEP::l_getCoordinateSystems(lua_State* L)
{
    lua_newtable(L);
	
    for(int i=0; i<=6; i++)
    {
	lua_pushinteger(L, i+1);
	lua_pushstring(L, nameOfCoordinateSystem( (CoordinateSystem)i ));
	lua_settable(L, -3);
    }

    return 1;
}

static void _tab2i(lua_State* L, int idx, int* vec)
{
    for(int i=0; i<2; i++)
    {
	lua_pushinteger(L, 1+i);
	lua_gettable(L, idx);
	if(lua_isnumber(L, -1))
	    vec[i] = lua_tointeger(L, -1) - 1 + i; // +1 on end range
	lua_pop(L, 1);
    }
}

// 
static void _getPSRange(lua_State* L, MEP* mep, int idx_p, int idx_s, int* point, int* site)
{
    site[0] = 0;
    site[1] = mep->numberOfSites();
    site[2] = (int)'t'; // we'll store info about the source type here. t = table, p = point

    point[0] = 0;
    point[1] = mep->numberOfImages();
    point[2] = (int)'t'; // we'll store info about the source type here.

    if(lua_isnumber(L, idx_p))
    {
	point[0] = lua_tointeger(L, idx_p) - 1;
	point[1] = point[0] + 1;
	point[2] = (int)'p'; // what a super ugly hack.
    }

    if(lua_istable(L, idx_p))
    {
	_tab2i(L, idx_p, point);
	point[2] = (int)'t';
    }
		
    if(lua_isnumber(L, idx_s))
    {
	site[0] = lua_tointeger(L, idx_s) - 1;
	site[1] = point[0] + 1;
	site[2] = (int)'p';
    }

    if(lua_istable(L, idx_s))
    {
	_tab2i(L, idx_s, site);
	site[2] = (int)'t';
    }
}

int MEP::l_setCoordinateSystem(lua_State* L, int idx)
{
    int point[2];
    int site[2];
	
    _getPSRange(L, this, idx+1, idx+2, point, site);

    const char* new_cs = lua_tostring(L, idx);

    CoordinateSystem newSystem = coordinateSystemByName(new_cs);

    if(newSystem != Undefined)
    {
	for(int p=point[0]; p<point[1]; p++)
	{
	    for(int s=site[0]; s<site[1]; s++)
	    {
		//printf("%i %i %s\n", p,s,nameOfCoordinateSystem(newSystem));
                VectorCS old = getPointSite(p, s);
                //print_vec(old);
                VectorCS conv = old.convertedToCoordinateSystem(newSystem);
                //print_vec(conv);
		setPointSite(p, s, conv);
	    }
	}
        energy_ok = false; // just in case
	return 0;
    }

    return luaL_error(L, "Unknown Coordinate System");
}

int MEP::l_getCoordinateSystem(lua_State* L)
{
    int point[3];
    int site[3];
	
    _getPSRange(L, this, 2, 3, point, site);

    if(point[2] == (int)'p' && site[2] == (int)'p')
    {
	lua_pushstring(L, nameOfCoordinateSystem( getPointSite(point[0], site[0]).cs ));
	return 1;
    }

    if(point[2] == (int)'t' && site[2] == (int)'p')
    {
	lua_newtable(L);

	int j = 1;
	for(int p=point[0]; p<point[1]; p++)
	{
	    lua_pushinteger(L, j); j++;
	    lua_pushstring(L, nameOfCoordinateSystem( getPointSite(p, site[0]).cs ));
	    lua_settable(L, -3);
	}

	return 1;
    }

    if(point[2] == (int)'p' && site[2] == (int)'t')
    {
	lua_newtable(L);

	int j = 1;
	for(int s=site[0]; s<site[1]; s++)
	{
	    lua_pushinteger(L, j); j++;
	    lua_pushstring(L, nameOfCoordinateSystem( getPointSite(point[0], s).cs ));
	    lua_settable(L, -3);
	}

	return 1;
    }


    if(point[2] == (int)'t' && site[2] == (int)'t')
    {
	lua_newtable(L);

	int j1 = 1;
	for(int p=point[0]; p<point[1]; p++)
	{
	    lua_pushinteger(L, j1); j1++;
	    lua_newtable(L);
	    int j2 = 1;
	    for(int s=site[0]; s<site[1]; s++)
	    {
		lua_pushinteger(L, j2); j2++;
		lua_pushstring(L, nameOfCoordinateSystem( getPointSite(p, s).cs ));
		lua_settable(L, -3);
	    }
	    lua_settable(L, -3);
	}
	return 1;
    }
	

    return 0;
}

void MEP::rotatePathAboutBy(const VectorCS& v, const double rad)
{
    VectorCS temp;
    VectorCS nv = v.convertedToCoordinateSystem(Cartesian).normalizedTo(1);

    const int np = numberOfImages();
    const int ns = numberOfSites();

    for(int p=0; p<np; p++)
    {
        for(int s=0; s<ns; s++)
        {
            getPointSite(p,s,temp);
            setPointSite(p,s,temp.rotatedAboutBy(nv, rad));
        }
    }
}

int  MEP::l_rotatePathAboutBy(lua_State* L, int base)
{
    double x1 = lua_tonumber(L, base+0);
    double x2 = lua_tonumber(L, base+1);
    double x3 = lua_tonumber(L, base+2);
    const char* cs_name = lua_tostring(L, base+3);
    double rad = lua_tonumber(L, base+4);

    VectorCS v(x1,x2,x3, coordinateSystemByName(cs_name));

    rotatePathAboutBy(v, rad);

    return 0;
}





double MEP::absoluteDifference(MEP* other, int point, double& max_diff, int& max_idx, vector<double>& vdiff)
{
    if(state_path.size() != other->state_path.size())
    {
	// printf("s1.size() = %i, s2.size() = %i\n", state_path.size(), other->state_path.size());
	return -1;
    }

    double d = 0;
    max_diff = 0;
    max_idx = 0;

    int ns = numberOfSites();
    int min = 0;
    int max = state_path.size();
   
    if(point >= 0)
    {
	min =  point    * ns;
	max = (point+1) * ns;
    }

    double q = 0;
    vdiff.clear();

    for(int i=min; i<max; i++)
    {
	const double diff = VectorCS::angleBetween(state_path[i], other->state_path[i]);
	// ABC
	// print_vec(state_path[i]);
	// print_vec(other->state_path[i]);

        q += diff;

        if(i % ns == ns-1)
        {
            vdiff.push_back(q);
            q = 0;
        }

	if(diff > max_diff)
	{
	    max_diff = diff;
	    max_idx = i;
	}
		
	d += diff;
    }

    return d;
}



void MEP::randomize(const double magnitude)
{
    const int num_sites = numberOfSites();
    double nv[3];

    const int ni = numberOfImages(); //images
    const int ns = numberOfSites();
	
    for(int i=0; i<ni; i++)
    {
	for(int s=0; s<ns; s++)
	{
	    const int k = (i*ns+s);

	    state_path[i].randomizeDirection(magnitude * image_site_mobility[i*ns + s]);
	}
    }
}



double MEP::distanceBetweenPoints(int p1, int p2, int site)
{
    //ABC
    const int ns = numberOfSites();

    return VectorCS::angleBetween(state_path[ns * p1 + site], state_path[ns * p2 + site]);
}

double MEP::distanceBetweenHyperPoints(int p1, int p2)
{
    vector<VectorCS> v1;
    vector<VectorCS> v2;
    
    getPoint(p1, v1);
    getPoint(p2, v2);
    

    double d = 0;
    const int ns = numberOfSites();
    for(int i=0; i<ns; i++)
    {
	d += VectorCS::angleBetween(v1[i], v2[i]);
    }
    return d;
}

void MEP::interpolateVectorCS(const vector<VectorCS>& v1, const vector<VectorCS>& v2, const double ratio, vector<VectorCS>& dest, bool backwards)
{
    for(unsigned int i=0; i<v1.size(); i++)
    {
	VectorCS t;
	interpolateVectorCS(v1[i], v2[i], ratio, t, backwards);
	dest.push_back(t);
    }
}

void MEP::interpolateVectorCS(const VectorCS& v1, const VectorCS& v2, const double ratio, VectorCS& dest, bool backwards)
{
    VectorCS nv1 = v1.normalizedTo(1).convertedToCoordinateSystem(Cartesian);
    VectorCS nv2 = v2.normalizedTo(1).convertedToCoordinateSystem(Cartesian);

    if(ratio == 0) //then no need to interpolate
    {
	dest = v1;
	return;
    }
    if(ratio == 1) 
    {
	dest = v2;
	return;
    }

    double a = VectorCS::angleBetween(nv1, nv2);

    if(backwards)
    {
        a = - (M_PI*2 - a);
    }
	
    VectorCS norm;

    if(fabs(a - 3.1415926538979) < 1e-8) //then colinear, need a random ortho vector
    {
	// notice the coord twiddle
	VectorCS t(-nv2.v[2], -nv2.v[0], nv2.v[1], Cartesian);

	norm = VectorCS::cross(nv1, t);

	// bad luck case:
	if(norm.magnitude() == 0) // then
	{
	    double* tt = t.v;
	    tt[0] = myrandf()*2.0-1.0;
	    tt[1] = myrandf()*2.0-1.0;
	    tt[2] = myrandf()*2.0-1.0;
	    norm = VectorCS::cross(nv1, t);
	}
    }
    else
    {
	norm = VectorCS::cross(nv1, nv2);
    }

    norm.setMagnitude(1);
	

    VectorCS res = nv1.rotatedAboutBy(norm, -a*ratio);
    res.setMagnitude( v1.magnitude() );

    if(ratio < 0.5)
	res.convertToCoordinateSystem(v1.cs);
    else
	res.convertToCoordinateSystem(v2.cs);
	
    dest = res;
}


void MEP::interpolatePoints(const int p1, const int p2, const int site, const double _ratio, vector<VectorCS>& dest, const double rjitter, bool backwards)
{
    const int num_sites = numberOfSites();

    const VectorCS& v1 = state_path[p1 * num_sites + site];
    const VectorCS& v2 = state_path[p2 * num_sites + site];
	
    VectorCS nv1 = v1.normalizedTo(1).convertedToCoordinateSystem(Cartesian);
    VectorCS nv2 = v2.normalizedTo(1).convertedToCoordinateSystem(Cartesian);

    double mobility = 1.0;
	
    const double m1 = getImageSiteMobility(p1, site);
    const double m2 = getImageSiteMobility(p2, site);
	
    if(m1 < mobility)
	mobility = m1;
    if(m2 < mobility)
	mobility = m2;
	
    double ratio = _ratio * (1.0 + rjitter * mobility);
	
    double a = VectorCS::angleBetween(nv1, nv2);

    if(backwards)
    {
        a = - (M_PI*2 - a);
    }

    if(ratio == 0) //then no need to interpolate
    {
	dest.push_back(v1.copy());
	return;
    }
	
    VectorCS norm;

    if(fabs(a - 3.1415926538979) < 1e-8) //then colinear, need a random ortho vector
    {
	// notice the coord twiddle
	VectorCS t(-nv2.v[2], -nv2.v[0], nv2.v[1], Cartesian);

	norm = VectorCS::cross(nv1, t);

	// bad luck case:
	if(norm.magnitude() == 0) // then
	{
	    double* tt = t.v;
	    tt[0] = myrandf()*2.0-1.0;
	    tt[1] = myrandf()*2.0-1.0;
	    tt[2] = myrandf()*2.0-1.0;
	    norm = VectorCS::cross(nv1, t);
	}
    }
    else
    {
	norm = VectorCS::cross(nv1, nv2);
    }

    norm.setMagnitude(1);
	

    VectorCS res = nv1.rotatedAboutBy(norm, -a*ratio);
	
    res.setMagnitude( v1.magnitude() );

    if(ratio < 0.5)
	res.convertToCoordinateSystem(v1.cs);
    else
	res.convertToCoordinateSystem(v2.cs);
	
    dest.push_back(res);
}



void MEP::interpolateHyperPoints(const int p1, const int p2, const double ratio, vector<VectorCS>& dest, const double jitter, bool* backwards)
{
    int n = numberOfSites();
    for(int i=0; i<n; i++)
    {
	double rjitter = (myrandf() * jitter*2 - jitter);
	interpolatePoints(p1,p2,i,ratio,dest, rjitter, backwards[i]);
    }
}

// get interval that bounds the distance
static int get_interval(const vector<double>& v, double x, double& ratio)
{
    for(unsigned int i=0; i<v.size(); i++)
    {
	const double y = v[i];
	if(x < y)
	{
	    ratio = x/y;
	    return i;
	}
	x -= y;
    }
	
    // clip to end
    ratio = 1;
    return v.size()-1;
}

void MEP::printState()
{
}

double MEP::calculateD12()
{
    if(good_distances)
	return d12_total;

    d12.clear();
    const int num_hyperpoints = numberOfImages();
    d12_total = 0;
    for(int i=0; i<num_hyperpoints-1; i++)
    {
	const double d = distanceBetweenHyperPoints(i,i+1);
	d12.push_back(d);
	d12_total += d;
    }
	
    good_distances = true;
    return d12_total;
}

int MEP::_swap(lua_State* L, int base)
{
    vector<int> src;
    vector<int> dest;

    if(!lua_istable(L, base) || !lua_istable(L, base+1))
	return luaL_error(L, "expected 2 tables in _swap");

    lua_pushnil(L);
    while(lua_next(L, base))
    {
	src.push_back(lua_tointeger(L, -1) - 1);
	lua_pop(L, 1);
    }

    lua_pushnil(L);
    while(lua_next(L, base+1))
    {
	dest.push_back(lua_tointeger(L, -1) - 1);
	lua_pop(L, 1);
    }

    const int ni = numberOfImages();
    for(unsigned int i=0; i<src.size() && i<dest.size(); i++)
    {
	const int a = src[i];
	const int b = dest[i];

	if(a < 0 || b < 0 || a >= ni || b >= ni)
	    return luaL_error(L, "_swap index out of bounds");
		
	VectorCS t = state_path[a];
	state_path[a] = state_path[b];
	state_path[b] = t;
    }

    return 0;
}

static void copyVectorTo(const vector<VectorCS>& src, vector<VectorCS>& dest)
{
    dest.clear();
    for(int i=0; i<src.size(); i++)
    {
	dest.push_back(src[i].copy());
    }
}

int MEP::_copy(lua_State* L, int base)
{
    vector<int> src;
    vector<int> dest;

    if(!lua_istable(L, base) || !lua_istable(L, base+1))
	return luaL_error(L, "expected 2 tables in _copy");

    lua_pushnil(L);
    while(lua_next(L, base))
    {
	src.push_back(lua_tointeger(L, -1) - 1);
	lua_pop(L, 1);
    }

    lua_pushnil(L);
    while(lua_next(L, base+1))
    {
	dest.push_back(lua_tointeger(L, -1) - 1);
	lua_pop(L, 1);
    }

    const int ni = numberOfImages();
    for(unsigned int i=0; i<src.size() && i<dest.size(); i++)
    {
	const int a = src[i];
	const int b = dest[i];

	if(a < 0 || b < 0 || a >= ni || b >= ni)
	    return luaL_error(L, "_copy index out of bounds");

	state_path[b] = state_path[a];
    }

    return 0;
}

int MEP::_resize(lua_State* L, int base)
{
    int ni = numberOfImages();
    int ns = numberOfSites();

    if(lua_isnumber(L, base))
	ni = lua_tointeger(L, base);

    if(lua_isnumber(L, base+1))
	ns = lua_tointeger(L, base+1);

    if(ns == 0)
    {
	sites.clear();
	state_path.clear();
	return 0;
    }

    sites.resize(ns*3, 0);

    state_path.resize(ni * numberOfSites());

    return 0;
}

#include <algorithm>    // std::sort
int MEP::resampleStatePath(lua_State* L)
{
    vector<double> points;
	
    double jitter = 0; //percent
    if(lua_isnumber(L, 3))
	jitter = lua_tonumber(L, 3);
	
    const int ns = numberOfSites();
    
    bool* backwards = new bool[ns];
    for(int i=0; i<ns; i++)
    {
        if(lua_istable(L, -1))
        {
            lua_pushinteger(L, i+1);
            lua_gettable(L, -2);
            backwards[i] = lua_toboolean(L, -1);
            lua_pop(L, 1);
        }
        else
            backwards[i] = false;
    }



    if(lua_isnumber(L, 2))
    {
	int num_points = lua_tointeger(L, 2);
	if(num_points < 2)
	    num_points = 2;
		
	for(int i=0; i<num_points; i++)
	    points.push_back(i);
    }
    else
    {
	if(lua_istable(L, 2))
	{
	    lua_pushnil(L);  /* first key */
	    while(lua_next(L, 2) != 0)
	    {
		points.push_back(lua_tonumber(L, -1));
		lua_pop(L, 1);
	    }
	}
	else
	{
	    return luaL_error(L, "resampleStatePath requires a number or a table of numbers");
	}
    }
	
	
    std::sort(points.begin(), points.end());

    // here we will add more and more points near existing points
    // until a goal number of points is hit
    if(lua_isnumber(L, 4))
    {
	int goal_pts = lua_tointeger(L, 4);
		
	int current_pts = points.size();
		
	int magic_number = current_pts * 2 - 2; //this is the number of bases+offsets
	// we'll grow in the positive direction from the 1st number, 
	// positive and negative directions from middle numbers
	// and negative direction from the end number
		
	double* base_offsets = new double[magic_number * 2];
	int j = 0;
	base_offsets[j+0] = points[0];
	base_offsets[j+1] = 0.25 * (points[1] - points[0]);
		
	j += 2;
	for(int i=1; i<current_pts-1; i++)
	{
	    base_offsets[j+0]   = points[i];
	    base_offsets[j+1] = 0.1 * (points[i-1] - points[i]);
	    base_offsets[j+2]   = points[i];
	    base_offsets[j+3] = 0.25 * (points[i+1] - points[i]);
	    j += 4;
	}
		
	base_offsets[j+0]   = points[current_pts-1];
	base_offsets[j+1] = 0.25 * (points[current_pts-2] - points[current_pts-1]);
		
	j = 0;
	while(points.size() < goal_pts)
	{
	    points.push_back( base_offsets[j*2+0] + base_offsets[j*2+1] );
	    base_offsets[j*2+1] *= 0.5;
	    j++;
			
	    if(j >= magic_number)
		j = 0;
	}
		
	delete [] base_offsets;

	std::sort(points.begin(), points.end());
    }
	
    const double max_point = points[ points.size() - 1];
    double distance = calculateD12(); //total distance
    vector<VectorCS> new_state_path;

    int new_num_images = points.size();
    // scaling points to same range as current points
    for(int i=0; i<new_num_images; i++)
    {
	points[i] *= (distance/max_point);
    }
	
    double ratio;
    for(int i=0; i<points.size(); i++)
    {
	int j = get_interval(d12, points[i], ratio);
	if(j != -1)
	{
// 			printf("range[ %i:%i]  ratio: %e\n", j, j+1, ratio); 
	    interpolateHyperPoints(j, j+1, ratio, new_state_path, jitter, backwards);
	}
	else
	{
	    int nos = numberOfSites();
	    for(int k=0; k<nos; k++)
	    {
		new_state_path.push_back(VectorCS());
	    }
	    /*
	      printf("failed to get interval\n");
	      printf("  total distance: %g\n", distance);
	      printf("  requested position: %g\n", points[i]);
	    */
	}
    }
	
    state_path.clear();
    for(unsigned int i=0; i<new_state_path.size(); i++)
    {
	state_path.push_back(new_state_path[i]);
    }
	
    // now we need to update mobility factors. The only case we will cover
    // is non-unitary endpoints.
    vector<double> first_mobility;// = image_site_mobility[0];
    vector<double> last_mobility;// = image_site_mobility[image_site_mobility.size()-1];
	
// 	new_num_images
	
    for(int i=0; i<ns; i++)
    {
	double d = 0;
	if(i < image_site_mobility.size())
	    d = image_site_mobility[i];
	first_mobility.push_back( d );
    }

    for(int i=image_site_mobility.size()-ns; i<image_site_mobility.size(); i++)
    {
	last_mobility.push_back( image_site_mobility[i] ); 
    }
	
    image_site_mobility.clear();
	
    for(int i=0; i<ns; i++)
    {
	image_site_mobility.push_back(first_mobility[i]);
    }
	
    for(int i=0; i<ns*(new_num_images-2); i++)
    {
	image_site_mobility.push_back(1.0);
    }
	
    for(int i=0; i<ns; i++)
    {
	double d = 0;
	if(i < image_site_mobility.size())
	    d = image_site_mobility[i];
	image_site_mobility.push_back(d);
    }
	
    energy_ok = false; //need to recalc energy
    good_distances = false;
    delete [] backwards;
    return 0;
}

void MEP::addSite(int x, int y, int z)
{
    sites.push_back(x);
    sites.push_back(y);
    sites.push_back(z);
    energy_ok = false; //need to recalc energy
}

//project gradients onto vector perpendicular to spin direction
void MEP::projForcePerpSpins(lua_State* L)
{
    if(force_vector.size() != state_path.size())
    {
	luaL_error(L, "forces are not computed or size mismatch");
	return;
    }

    for(unsigned int i=0; i<state_path.size(); i++)
    {
	force_vector[i] = force_vector[i].rejected( state_path[i] );
    }
}

int MEP::applyForces(lua_State* L)
{
    if(force_vector.size() != state_path.size())
	return luaL_error(L, "forces are not computed or size mismatch");
	
    bool use_mobility = true;
    if(lua_isboolean(L, 2))
	use_mobility = lua_toboolean(L, 2);

    bool reject_path = true; // vector rejection against path


    const int num_sites = numberOfSites();
	
    const int np = numberOfImages();
    const int ns = numberOfSites();
	
    // compute cartesian path tangent
    path_tangent.clear();
    path_tangent.resize(state_path.size());
    computeTangent(0, 1, 0);
    for(int i=1; i<np-1; i++)
    {
	computeTangent(i-1,i+1,i);
    }
    computeTangent(np-2, np-1, np-1);


    vector<VectorCS> new_state_path(state_path);

    // end points are no longer hardcoded fixed. 
    for(int k=0; k<state_path.size(); k++)
    {
	const double mag = state_path[k].magnitude();

	if(use_mobility)
	    new_state_path[k] = VectorCS::axpy(-image_site_mobility[k]*beta, force_vector[k], state_path[k]);
	else
	    new_state_path[k] = VectorCS::axpy(-beta, force_vector[k], state_path[k]);

	new_state_path[k].setMagnitude(mag);
    }

    vector<VectorCS> cart_diff;
    for(int k=0; k<state_path.size(); k++)
    {
	const VectorCS& new_ = new_state_path[k];
	const VectorCS& old_ = state_path[k];
	
	VectorCS v = VectorCS::axpy(-1, 
				    old_.convertedToCoordinateSystem(Cartesian), 
				    new_.convertedToCoordinateSystem(Cartesian));
	cart_diff.push_back(v);
    }

    // now we have the vector from the old state to the new state, we can 
    // reject against path tangent now
    double* a = new double[ns*3];
    double* b = new double[ns*3];
    for(int p=0; p>np; p++)
    {
	// considering point as a high dimension vector
	vector<VectorCS> ptan;
	getPathTangentPoint(p, ptan);
	
	// get the displacement vector and tangent in array form
	for(int i=0; i<ns; i++)
	{
	    a[i*3+0] = cart_diff[p*ns + i].v[0];
	    a[i*3+1] = cart_diff[p*ns + i].v[1];
	    a[i*3+2] = cart_diff[p*ns + i].v[2];

	    a[i*3+0] = path_tangent[p*ns + i].v[0];
	    b[i*3+1] = path_tangent[p*ns + i].v[1];
	    b[i*3+2] = path_tangent[p*ns + i].v[2];
	}


	double ab = 0;
	double bb = 0;

	for(int i=0; i<ns*3; i++)
	{
	    ab += a[i]*b[i];
	    bb += b[i]*b[i];
	}

	if(bb == 0)
	    bb = 1;

	for(int i=0; i<ns*3; i++)
	{
	    a[i] = a[i] - (ab/bb) * b[i];
	}

	for(int i=0; i<ns; i++)
	{
	    cart_diff[p*ns + i].v[0] = a[i*3+0];
	    cart_diff[p*ns + i].v[1] = a[i*3+0];
	    cart_diff[p*ns + i].v[2] = a[i*3+0];
	}
    }

    delete [] a;
    delete [] b;

    // cart_diff now has vector rejected displacement
    // time to add it to the initial point
    for(int k=0; k<state_path.size(); k++)
    {
	const double mag = state_path[k].magnitude();

	state_path[k] = VectorCS::axpy(1.0, state_path[k].convertedToCoordinateSystem(Cartesian),
				       cart_diff[k]).convertedToCoordinateSystem(state_path[k].cs);

	state_path[k].setMagnitude(mag);
    }

#if 0
    // end points are no longer hardcoded fixed. 
    for(int k=0; k<state_path.size(); k++)
    {
	const double mag = state_path[k].magnitude();

	if(use_mobility)
	    state_path[k] = VectorCS::axpy(-image_site_mobility[k]*beta, force_vector[k], state_path[k]);
	else
	    state_path[k] = VectorCS::axpy(-beta, force_vector[k], state_path[k]);

	state_path[k].setMagnitude(mag);
    }
#endif
    return 0;
}
	

void MEP::computeTangent(const int p1, const int p2, const int dest)
{
    VectorCS a;
    VectorCS b;
    VectorCS c;

    double sum2 = 0;

    const int ns = numberOfSites();
	

    // compute difference
    for(int s=0; s<ns; s++)
    {
	a = getPointSite(p1, s).convertedToCoordinateSystem(Cartesian);
	b = getPointSite(p2, s).convertedToCoordinateSystem(Cartesian);

	c = VectorCS::axpy(-1.0, b, a);
	sum2 += c.magnitude()*c.magnitude();

	path_tangent[dest*ns + s] = c;
    }

    if(sum2 == 0)
	sum2 = 1.0;

    for(int s=0; s<ns; s++)
    {
	path_tangent[dest*ns + s].scale(1/sqrt(sum2));		
    }
	

}


//project gradients onto vector perpendicular to path direction
// the plan is to compute vectors along the path (cartesian), apply the forces to the state
// saving in a temporary buffer. Convert old and new states to cartesian and get the difference
// from that we can do the proper vector rejection and move that back into the force
void MEP::projForcePerpPath(lua_State* L) //, int get_index, int set_index, int energy_index) 
{
    //for(int k=0; k<state_path.size(); k++)
    //    print_vec("before", force_vector[k]);

    path_tangent.clear();
    path_tangent.resize(state_path.size());

    const int ns = numberOfSites();
    const int np = numberOfImages();

    computeTangent(0, 1, 0);
    for(int i=1; i<np-1; i++)
    {
	computeTangent(i-1,i+1,i);
    }
    computeTangent(np-2, np-1, np-1);


    // new state is where the force would take things if applied as is
    vector<VectorCS> new_state;
    for(int k=0; k<state_path.size(); k++)
    {
	const double mag = state_path[k].magnitude();
	VectorCS v = VectorCS::axpy(-image_site_mobility[k]*beta, force_vector[k], state_path[k]);
	v.setMagnitude(mag);
	v.convertToCoordinateSystem(Cartesian);
	new_state.push_back(v);
    }

    vector<VectorCS> cart_diff;
    for(int k=0; k<state_path.size(); k++)
    {
	const VectorCS& new_ = new_state[k];
	const VectorCS& old_ = state_path[k];
	
	VectorCS v = VectorCS::axpy(-1, old_.convertedToCoordinateSystem(Cartesian), new_);
	cart_diff.push_back(v);
    }

    // now we have a cartesian vector from the old state to the results of the old + beta * force
    // so we can do vector rejection properly
    double* a = new double[ns*3];
    double* b = new double[ns*3];
    for(int p=0; p>np; p++)
    {
	// considering point as a high dimension vector
	vector<VectorCS> ptan;
	getPathTangentPoint(p, ptan);
	
	// get the displacement vector and tangent in array form
	for(int i=0; i<ns; i++)
	{
	    a[i*3+0] = cart_diff[p*ns + i].v[0];
	    a[i*3+1] = cart_diff[p*ns + i].v[1];
	    a[i*3+2] = cart_diff[p*ns + i].v[2];

	    a[i*3+0] = path_tangent[p*ns + i].v[0];
	    b[i*3+1] = path_tangent[p*ns + i].v[1];
	    b[i*3+2] = path_tangent[p*ns + i].v[2];
	}


	double ab = 0;
	double bb = 0;

	for(int i=0; i<ns*3; i++)
	{
	    ab += a[i]*b[i];
	    bb += b[i]*b[i];
	}

	if(bb == 0)
	    bb = 1;

	for(int i=0; i<ns*3; i++)
	{
	    a[i] = a[i] - (ab/bb) * b[i];
	}

	for(int i=0; i<ns; i++)
	{
	    cart_diff[p*ns + i].v[0] = a[i*3+0];
	    cart_diff[p*ns + i].v[1] = a[i*3+0];
	    cart_diff[p*ns + i].v[2] = a[i*3+0];
	}
    }

    delete [] a;
    delete [] b;

    // now we add the rejected cart diff to the initial point to get a rejected new point
    new_state.clear();
    for(int k=0; k<state_path.size(); k++)
    {
	const double mag = state_path[k].magnitude();
	VectorCS v = VectorCS::axpy(1, 
                                    cart_diff[k], 
                                    state_path[k].convertedToCoordinateSystem(Cartesian));
	v.setMagnitude(mag);
	new_state.push_back(v.convertedToCoordinateSystem(state_path[k].cs));
    }

    // the difference between the new point and the current point is beta * force
    force_vector.clear();
    for(int k=0; k<state_path.size(); k++)
    {
        force_vector.push_back(VectorCS::axpy(-1, new_state[k], state_path[k]));
        force_vector[k].scale(1/beta);
    }

    //for(int k=0; k<state_path.size(); k++)
    //    print_vec("after", force_vector[k]);

    if(false)
    {
	int* i = (int*)5;
	*i = 5; //crash
    }
}



//project gradients onto vector perpendicular to path direction
void MEP::projForcePath(lua_State* L) //, int get_index, int set_index, int energy_index) 
{
    path_tangent.clear();
    path_tangent.resize(state_path.size());

    const int ns = numberOfSites();
    const int np = numberOfImages();

    computeTangent(0, 1, 0);
    for(int i=1; i<np-1; i++)
    {
	computeTangent(i-1,i+1,i);
    }
    computeTangent(np-2, np-1, np-1);

    // Cartesian values
    // double* proj_c = new double[ss];
    // double* force_c = new double[ss];
    // double* v_c = new double[ss];

    for(int p=0; p<np; p++)
    {
	vector<VectorCS> force;
	vector<VectorCS> force_orig;
	getForcePoint(p, force_orig);

	vector<VectorCS> ptan;
	getPathTangentPoint(p, ptan);
		
	for(int s=0; s<ns; s++)
	{
	    force.push_back(force_orig[s].convertedToCoordinateSystem(Cartesian));
	    ptan[s].convertToCoordinateSystem(Cartesian);
	}

	const double bb = cart_dot(ptan, ptan); //dot(v_c, v_c, ss);
	if(bb == 0)
	{
	    for(int s=0; s<ns; s++)
		force[s] = VectorCS(0,0,0, Cartesian);
	}
	else
	{
	    const double ab = cart_dot(force, ptan);
	    for(int s=0; s<ns; s++)
		force[s] = ptan[s].scaled(ab/bb);
	}

	for(int s=0; s<ns; s++)
	    force[s].convertToCoordinateSystem( force_orig[s].cs );

	setForcePoint(p, force );
    }

}

// back up old sites so we can restore after
void MEP::saveConfiguration(vector<double>& buffer)
{
    buffer.clear();
    const int ns = numberOfSites();
	
    for(int s=0; s<ns; s++)
    {
	int* site = &sites[s*3];
	//printf(">> %i %i %i\n", site[0], site[1], site[2]);
	getSiteSpin(site, buffer);
    }
}

// restore state
void MEP::loadConfiguration(vector<double>& buffer)
{
    const int num_sites = numberOfSites();

    for(int s=0; s<num_sites; s++)
    {
	setSiteSpin(&sites[s*3], &buffer[s*3]);
    }
}

void MEP::getSiteSpin(int* site3, VectorCS& m3)
{
    double t[3];

    /*
    lua_pushvalue(L, get_index);
    lua_pushinteger(L, site3[0]+1);
    lua_pushinteger(L, site3[1]+1);
    lua_pushinteger(L, site3[2]+1);
    lua_call(L, 3, 3);

    t[0] = lua_tonumber(L, -3);
    t[1] = lua_tonumber(L, -2);
    t[2] = lua_tonumber(L, -1);

    lua_pop(L, 3);	
    */
    get_site_ss(site3[0], site3[1], site3[2], t);

    VectorCS vec(t, Cartesian);
    vec.convertToCoordinateSystem(m3.cs);

    m3.v[0] = vec.v[0];
    m3.v[1] = vec.v[1];
    m3.v[2] = vec.v[2];
}

void MEP::getSiteSpin(int* site3, vector<double>& v)
{
    VectorCS m(0,0,0,Cartesian);
    getSiteSpin(site3, m); // will get in Cartesian
    v.push_back(m.v[0]);
    v.push_back(m.v[1]);
    v.push_back(m.v[2]);
}


double MEP::getImageSiteMobility(const int image, const int site)
{
    int k = image * numberOfSites() + site;

    if(k < 0 || k >= image_site_mobility.size())
    {
	printf("bad index in getImageSiteMobility\n");
	return 0;
    }
	
    return image_site_mobility[k];
}


void MEP::setImageSiteMobility(const int image, const int site, double mobility)
{
    int k = image * numberOfSites() + site;
    if(k < 0)
    {
	printf("bad index in setImageSiteMobility\n");
	return;
    }
    while(image_site_mobility.size() <= k)
    {
	image_site_mobility.push_back(1.0);
    }
	
    image_site_mobility[k] = mobility;
}

void MEP::setSiteSpin(int* site3, const double* _mm)
{
    VectorCS mm(_mm[0], _mm[1], _mm[2], Cartesian);
    setSiteSpin(site3, mm);
}

void MEP::setSiteSpin(int* site3, const VectorCS& mm)
{
    VectorCS tt = mm.convertedToCoordinateSystem(Cartesian);
    double* t = tt.v;

    if( set_site_ss(site3[0], site3[1], site3[2], t) )
    {
	energy_ok = false; //need to recalc energy
	good_distances = false;
    }
}

void MEP::setAllSpins(const vector<VectorCS>& m)
{
    int num_sites = numberOfSites();
    for(int s=0; s<num_sites; s++)
	setSiteSpin(&sites[s*3], m[s]);
}

void MEP::getAllSpins(vector<VectorCS>& m)
{
    int num_sites = numberOfSites();
    for(int s=0; s<num_sites; s++)
	getSiteSpin(&sites[s*3], m[s]);
}

double MEP::getEnergy(lua_State* L, const vector<VectorCS>& v)
{
    if(fitEnergy)
        return fitEnergy->eval(v);

    setAllSpins(v);
    return get_energy_ss(L);
}


double MEP::getEnergy(lua_State* L)
{
    return get_energy_ss(L);
#if 0
    lua_pushvalue(L, energy_index);
    lua_call(L, 0, 1);
    double e = lua_tonumber(L, -1);
    lua_pop(L, 1);
    return e;
#endif
}

double MEP::energyOfCustomPoint(const vector<VectorCS>& v1)
{
    if(fitEnergy)
    {
        return fitEnergy->eval(v1);
    }

    const int ns = numberOfSites();
    if(v1.size() != ns)
        return 0;

    vector<double> cfg;
    saveConfiguration(cfg);
   
    for(int s=0; s<ns; s++)
    {
        setSiteSpin(&sites[s*3], v1[s]);
    }
    

    double e = getEnergy(L);

    loadConfiguration(cfg);

    return e;
}


int MEP::calculateEnergies(lua_State* L)
{
    if(energy_ok)
	return 0;
    energies.clear();
    const int ns = numberOfSites();
    const int np = numberOfImages();

    if(fitEnergy) // we can do it the easy way
    {
        for(int p=0; p<np; p++)
        {
            vector<VectorCS> vec;
            getPoint(p, vec);
            energies.push_back( fitEnergy->eval(vec) );
        }

        energy_ok = true;
        return 0;
    }


    // back up old sites so we can restore after
    vector<double> cfg;
    saveConfiguration(cfg);

    for(int p=0; p<np; p++)
    {
	for(int s=0; s<ns; s++)
	{
	    setSiteSpin(&sites[s*3], state_path[p * ns + s]);
	}

	energies.push_back(getEnergy(L)); 
    }

    loadConfiguration(cfg);
	
    energy_ok = true;
    return 0;
}

int MEP::calculateSingleEnergy(lua_State* L)
{
    const int ns = numberOfSites();
    const int np = numberOfImages();

    int p = lua_tonumber(L, 2) - 1;

    if(p<0 || p>=np)
        return luaL_error(L, "invalid point");

    if(fitEnergy) // then we do it the easy way
    {
        vector<VectorCS> vec;
        getPoint(p, vec);
        lua_pushnumber(L, fitEnergy->eval(vec) );
        return 1;
    }

    // back up old sites so we can restore after
    vector<double> cfg;
    saveConfiguration(cfg);

    for(int s=0; s<ns; s++)
    {
        setSiteSpin(&sites[s*3], state_path[p * ns + s]);
    }

    lua_pushnumber(L, getEnergy(L));
    loadConfiguration(cfg);
	
    return 1;
}

double MEP::rightDeriv(vector<double>& src, int i)
{
    double h = 0;
	
    double valHere = src[i];
	
    while(h == 0)
    {
	if(i >= src.size())
	    return 0;
		
	h += d12[i];
	i++;
    }
	
    return ( src[i] - valHere ) / h;
}

double MEP::leftDeriv(vector<double>& src, int i)
{
    double h = 0;
	
    double valHere = src[i];
	
    while(h == 0)
    {
	if(i < 0)
	    return 0;
		
	h += d12[i-1];
	i--;
    }
	
    return ( valHere - src[i]) / h;
}
	
void MEP::listDerivN(vector<double>& dest, vector<double>& src)
{
    dest.clear();
    if(d12_total == 0)
    {
	for(int i=0; i<src.size(); i++)
	    dest.push_back(0);
    }
    else
    {
	dest.push_back( rightDeriv(src, 0) );
	const int num_hyperpoints = d12.size() + 1;
	for(int i=1; i<num_hyperpoints-1; i++)
	{
	    const double deLeft  = leftDeriv(src, i);
	    const double deRight = rightDeriv(src, i);
	    dest.push_back( 0.5 * (deLeft + deRight) );
	}
	dest.push_back( leftDeriv(src, num_hyperpoints-1) );
    }
}


int MEP::calculatePathEnergyNDeriv(lua_State* L, int n, vector<double>& nderiv)
{
    // populate energies vector
    calculateEnergies(L); //always returns 0

    double distance = calculateD12(); //total distance
    nderiv.clear();
	
    if(n < 1)
    {
	for(unsigned int i=0; i<energies.size(); i++)
	{
	    nderiv.push_back(energies[i]);
	}
	return 0;
    }
	
    if(n == 1)
    {
	listDerivN(nderiv, energies);
	return 0;
    }
	
    // n > 1
    vector<double> nmoDeriv; //n-1 deriv	
	
    calculatePathEnergyNDeriv(L, n-1, nmoDeriv);

    listDerivN(nderiv, nmoDeriv);
	
    return 0;
}



static bool vectorElementChange(vector<VectorCS>& vec, int c, double change, const bool fixedRadius)
{
    int i = c % 3;
    int j = (c - i) / 3;

    if(fixedRadius)
    {
	if(vec[j].isType(Spherical) || vec[j].isType(Canonical))
	    if(i == 0) // zero element is R
		return false; // no change
    }

    vec[j].v[i] += change;
    return true; // real change
}



static void rescale_vectors(vector<VectorCS>& vecs, vector<double>& mags)
{
    for(int i=0; i<vecs.size(); i++)
    {
	vecs[i].setMagnitude(mags[i]);
    }
}

double MEP::computePointSecondDerivativeAB(lua_State* L, int p, int c1, int c2, double _dc1, double _dc2)
{
    // back up old sites so we can restore after
    vector<double> cfg;
    saveConfiguration(cfg);

    const int num_sites = numberOfSites();
    vector<VectorCS> vec;
    getPoint(p, vec);

    double result;

    double e1,e2,e3,e4;
    double d1,d2;

    double stepSize1[3];
    double stepSize2[3];

    const int site1 = (c1 - (c1%3)) / 3;
    const int site2 = (c2 - (c2%3)) / 3;

    vec[site1].stepSize(epsilon, stepSize1);
    vec[site2].stepSize(epsilon, stepSize2);

    double dx1 = stepSize1[c1 % 3];
    double dx2 = stepSize2[c2 % 3];
	
    if(_dc1 > 0)
	dx1 = _dc1;
    if(_dc2 > 0)
	dx2 = _dc2;

    // calc upper deriv energies
    e1 = 0;
    if(vectorElementChange(vec, c1, dx1, fixedRadius) &&
       vectorElementChange(vec, c2, dx2, fixedRadius) )
    {
        setAllSpins(vec);
        e1 = getEnergy(L, vec);
        getPoint(p, vec); // to reset
    }

	
    e2 = 0;
    if(vectorElementChange(vec, c1, dx1, fixedRadius) &&
       vectorElementChange(vec, c2,-dx2, fixedRadius) )
    {
        setAllSpins(vec);
        e2 = getEnergy(L, vec);
        getPoint(p, vec);
    }

    // calc lower deriv energies
    e3 = 0;
    if(vectorElementChange(vec, c1,-dx1, fixedRadius) &&
       vectorElementChange(vec, c2, dx2, fixedRadius))
    {
        setAllSpins(vec);
        e3 = getEnergy(L, vec);
        getPoint(p, vec);
    }

    e4 = 0;
    if(vectorElementChange(vec, c1,-dx1, fixedRadius) &&
       vectorElementChange(vec, c2,-dx2, fixedRadius))
    {
        setAllSpins(vec);
        e4 = getEnergy(L, vec);
        getPoint(p, vec);
    }

    double diff_e1_e2 = (e1 - e2);
    double diff_e3_e4 = (e3 - e4);

    const double dd1 = diff_e1_e2 / (2.0 * dx2);
    const double dd2 = diff_e3_e4 / (2.0 * dx2);

    result = (dd1 - dd2) / (2.0 * dx1);

    // restore existing cfg
    loadConfiguration(cfg);	

    return result;
}


// partial derivs
void MEP::computePointSecondDerivative(lua_State* L, int p, double* derivsAB)
{
    const int num_sites = numberOfSites();
    int deriv_pos = 0;
    for(int c1=0; c1<num_sites*3; c1++)
    {
	for(int c2=0; c2<num_sites*3; c2++)
	{
	    derivsAB[deriv_pos] = computePointSecondDerivativeAB(L, p, c1, c2);
	    deriv_pos++;
	}
    }
}

void MEP::getPoint(int p, vector<VectorCS>& dest)
{
    dest.clear();
    const int ns = numberOfSites();
    int max = state_path.size();

    for(int i=0; i<ns; i++)
    {
        const int idx = p * ns + i;

        if(idx >= max)
            printf("%i %i\n", idx, max);

	dest.push_back( state_path[idx].copy() );
    }
}

void MEP::setPoint(int p, vector<VectorCS>& src)
{
    const int ns = numberOfSites();
    for(int i=0; i<ns; i++)
    {
	state_path[p * ns + i] = src[i];
    }	
}

void MEP::getPathTangentPoint(int p, vector<VectorCS>& dest)
{
    dest.clear();
    const int ns = numberOfSites();
    for(int i=0; i<ns; i++)
    {
	dest.push_back( path_tangent[p * ns + i] );
    }
}

void MEP::setPathTangentPoint(int p, vector<VectorCS>& src)
{
    const int ns = numberOfSites();
    for(int i=0; i<ns; i++)
    {
	path_tangent[p * ns + i] = src[i].copy();
    }	
}



void MEP::getForcePoint(int p, vector<VectorCS>& dest)
{
    dest.clear();
    const int ns = numberOfSites();
    for(int i=0; i<ns; i++)
    {
	dest.push_back( force_vector[p * ns + i] );
    }
}

void MEP::setForcePoint(int p, vector<VectorCS>& src)
{
    const int ns = numberOfSites();
    for(int i=0; i<ns; i++)
    {
	force_vector[p * ns + i] = src[i].copy();
    }	
}

void MEP::getPointSite(int p, int s, VectorCS& dest)
{
    const int ns = numberOfSites();
    dest = state_path[p * ns + s].copy();
}
void MEP::setPointSite(int p, int s, VectorCS src)
{
    if(p < 0 || s < 0 || p >= numberOfImages() || s >= numberOfSites())
	return;
    const int ns = numberOfSites();
    state_path[p * ns + s] = src;

    energy_ok = false;
    good_distances = false;

}

CoordinateSystem MEP::getCSAt(int p, int s)
{
    if(p < 0 || p >= numberOfImages())
	return Cartesian;
    if(s < 0 || s >= numberOfSites())
	return Cartesian;
    VectorCS v;
    getPointSite(p,s,v);
    return v.cs;
}



void MEP::computePointFirstDerivative(lua_State* L, int p, const double p_energy, vector<VectorCS>& d)
{
    d.clear();
    const int num_sites = numberOfSites();
    for(int c=0; c<num_sites; c++)
    {
	double v1 = computePointFirstDerivativeC(L, p, p_energy, c*3+0);
	double v2 = computePointFirstDerivativeC(L, p, p_energy, c*3+1);
	double v3 = computePointFirstDerivativeC(L, p, p_energy, c*3+2);

	d.push_back(VectorCS(v1,v2,v3, getPointSite(p,c).cs));
    }
}



void MEP::computeVecFirstDerivative(lua_State* L, vector<VectorCS>& vec, const double vec_energy, vector<VectorCS>& d)
{
    d.clear();
    const int num_sites = numberOfSites();
    for(int c=0; c<num_sites; c++)
    {
	double x1 = computeVecFirstDerivativeC(L, vec, vec_energy, c*3+0);
	double x2 = computeVecFirstDerivativeC(L, vec, vec_energy, c*3+1);
	double x3 = computeVecFirstDerivativeC(L, vec, vec_energy, c*3+2);
		
	d.push_back(VectorCS(x1,x2,x3, vec[c].cs));
    }
}

/*
  void print_vec(double* v, int n)
  {
  for(int i=0; i<n; i++)
  {
  printf("%g", v[i]);
  if(i+1<n)
  printf("\t");
  else
  printf("\n");
  }
  }
*/

double MEP::vecEnergy(lua_State* L, vector<VectorCS>& vec)
{
    if(fitEnergy)
        return getEnergy(L, vec);

    // back up old sites so we can restore after
    vector<double> cfg;
    saveConfiguration(cfg);

    setAllSpins(vec);
    double e = getEnergy(L);

    loadConfiguration(cfg); // restore
    return e;
}

double MEP::computeVecFirstDerivativeC(lua_State* L, vector<VectorCS>& vec, const double energy, int c1)
{
    // back up old sites so we can restore after
    vector<double> cfg;
    saveConfiguration(cfg);
    const int num_sites = numberOfSites();
    double result;

    vector<VectorCS> state;
    vector<double> mags;

    double e1,e2,e3,e4,e5;
    double d1;

    const int c1m3 = c1 % 3;

    const int site1 = (c1 - c1m3) / 3;

    double scaleFactors1[3];

    double stepSize1[3];

    vec[site1].stepSize(epsilon, stepSize1);

    const double dx1 = stepSize1[c1m3]; // / sf;

    for(int i=0; i<num_sites; i++)
    {
	mags.push_back( vec[i].magnitude() );
    }


    copyVectorTo(vec, state);
    vectorElementChange(state, c1,-dx1, fixedRadius);
    rescale_vectors(state, mags);
    e1 = getEnergy(L, state);

    e2 = energy;

    copyVectorTo(vec, state);
    vectorElementChange(state, c1, dx1, fixedRadius);
    rescale_vectors(state, mags);
    setAllSpins(state);
    e3 = getEnergy(L, state);

    double v1 = (e3 - e2) / dx1;
    double v2 = (e2 - e1) / dx1;

    result = (v1 + v2) * 0.5;

    loadConfiguration(cfg);	

    return result;
}



double MEP::computePointFirstDerivativeC(lua_State* L, int p, const double p_energy, int c1)
{
    vector<VectorCS> vec;
    getPoint(p, vec);
    return computeVecFirstDerivativeC(L, vec, p_energy, c1);
}




#include "mep_spheres.h"

// used to replace arbitrarily nested for loops
static bool v_inc(vector<int>& state, vector<int>& max, int pos=0)
{
    if(pos >= state.size())
        return false;
    state[pos]++;
    if(state[pos] == max[pos])
    {
        state[pos] = 0;
        return v_inc(state, max, pos+1);
    }
    return true;
}

typedef struct NSpherePoint
{
    vector<int> idx;
    vector<VectorCS> vvec;
    vector<NSpherePoint*> neighbours;
    double grad;
    double en;
} NSpherePoint;

static void build_veccs(const sphere* Sphere, vector<int>& idx, vector<double>& mag, vector<VectorCS>& vv)
{
    vv.clear();
    for(int i=0; i<mag.size(); i++)
    {
        const sphere& s = Sphere[ idx[i] ];
        VectorCS v( s.vertex, Cartesian);
        v.convertToCoordinateSystem(Spherical);
        v.setMagnitude( mag[i] );
        vv.push_back(v);
    }
}

static bool same_vint(vector<int>& a, vector<int>& b)
{
    if(a.size() != b.size())
        return false;

    for(int i=0; i<a.size(); i++)
        if(a[i] != b[i])
            return false;

    return true;
}

static int num_neighbours(const int* neighbours)
{
    int n = 0;
    for(;neighbours[n] >= 0; n++);
    return n;
}

int MEP::findCriticalIslands(lua_State* L)
{
    int n = lua_tointeger(L, 2); // 1 is mep

    const int ns = numberOfSites();
    const sphere* Sphere = get_sphere(n);
    if(Sphere == 0)
        return luaL_error(L, "Bad subdivision number");

    vector<VectorCS> vec;
    vector<VectorCS> grad;

    getPoint(1, vec); // any point, just looking for magnitudes
    vector<double> mags;
    for(int i=0; i<ns; i++)
    {
	mags.push_back( vec[i].magnitude() );
    }

    // determine the number of verts on the sphere
    int num_verts = 0;
    for(;Sphere[num_verts].neighbours; num_verts++);


    vector<NSpherePoint*> points; //hyper-points 

    
    vector<int> state;
    vector<int> vmax;

    for(int j=0; j<ns; j++)
    {
        state.push_back(0);
        vmax.push_back(num_verts);
    }

    do
    {
        NSpherePoint* p = new NSpherePoint;
        for(int j=0; j<ns; j++)
            p->idx.push_back(state[j]);

        build_veccs(Sphere, p->idx, mags, p->vvec);
        points.push_back(p);
    }while(v_inc(state, vmax));


    // now have a flat list of all hyper points
    // time to calculate some values
    for(int i=0; i<points.size(); i++)
    {
        NSpherePoint* p = points[i];

        double vec_energy = vecEnergy(L, p->vvec);

        computeVecFirstDerivative(L, p->vvec, vec_energy, grad);
        for(int j=0; j<ns; j++)
            grad[j].zeroRadialComponent();

        double current_grad = sqrt(cart_norm2(grad));
        p->grad = current_grad;
        p->en = vec_energy;
        //print_vecv(vec);
        //printf("%6d %6d %e\n", i, points.size(), current_grad);
    }

    // ok, now we want to build a more convenient neighbour list
    // to reduce overall neighbour searches
    for(int i=0; i<points.size(); i++)
    {
        state.clear();
        vmax.clear();

        // idx are the indices into the Spheres
        vector<int>& idx = points[i]->idx;;
        vector<const int*> neighbour_lists;

        for(int j=0; j<ns; j++)
        {
            int p = idx[j];
            state.push_back(0);
            neighbour_lists.push_back( Sphere[p].neighbours );
            vmax.push_back(num_neighbours(neighbour_lists[j]));
        }

        do
        {
            vector<int> nn; //decode state to neighbour
            for(int j=0; j<state.size(); j++)
            {
                nn.push_back( neighbour_lists[j][state[j]] );
            }

            // nn now holds the indices of one of the neighbours of this hyperpoint
            // we will look through all the points to turn this into a pointer
            for(int k=0; k<points.size(); k++)
            {
                if(same_vint(nn, points[k]->idx))
                {
                    points[i]->neighbours.push_back( points[k] );
                }
            }
        }while(v_inc(state, vmax));
    }


    lua_newtable(L);
    int min_tab = lua_gettop(L);

    lua_newtable(L);
    int max_tab = lua_gettop(L);

    lua_newtable(L);
    int flat_tab = lua_gettop(L);

    lua_newtable(L);
    int all_tab = lua_gettop(L);

    int t_min_idx = 1;
    int t_max_idx = 1;
    int t_flat_idx = 1;
    int t_all_idx = 1;
    // at this point we should have a list of NSpherePoints with populated grad and neighbours
    // time to look for points with grads lower than all neighbours
    for(int i=0; i<points.size(); i++)
    {
        bool ok = true;
        double en_here = points[i]->en;

        bool is_min = true;
        bool is_max = true;
        bool is_sad = true;

        for(int j=0; j<points[i]->neighbours.size(); j++)
        {
            if(points[i]->neighbours[j]->en < en_here)
                is_min = false;
        }

        for(int j=0; j<points[i]->neighbours.size(); j++)
        {
            if(points[i]->neighbours[j]->en > en_here)
                is_max = false;
        }


        if(ok)
        {
            vector<VectorCS>& vec = points[i]->vvec;

            lua_pushinteger(L, t_min_idx); t_min_idx++;
            lua_pushVVectorCS(L, vec);
            lua_settable(L, min_tab);

            lua_pushinteger(L, t_all_idx); t_all_idx++;
            lua_pushVVectorCS(L, vec);
            lua_settable(L, all_tab);
        }



        // checking for high energy island
        ok = true;
        for(int j=0; j<points[i]->neighbours.size(); j++)
        {
            if(points[i]->neighbours[j]->en > en_here)
                ok = false;
        }
        if(ok)
        {
            vector<VectorCS>& vec = points[i]->vvec;

            lua_pushinteger(L, t_max_idx); t_max_idx++;
            lua_pushVVectorCS(L, vec);
            lua_settable(L, max_tab);

            lua_pushinteger(L, t_all_idx); t_all_idx++;
            lua_pushVVectorCS(L, vec);
            lua_settable(L, all_tab);
        }




        // checking for low grad island
        ok = true;
        double g = points[i]->grad;

        
        for(int j=0; j<points[i]->neighbours.size(); j++)
        {
            if(points[i]->neighbours[j]->grad < g)
                ok = false;
        }

        if(ok)
        {
            vector<VectorCS>& vec = points[i]->vvec;

            lua_pushinteger(L, t_flat_idx); t_flat_idx++;
            lua_pushVVectorCS(L, vec);
            lua_settable(L, flat_tab);

            lua_pushinteger(L, t_all_idx); t_all_idx++;
            lua_pushVVectorCS(L, vec);
            lua_settable(L, all_tab);
        }
    }

    for(int i=0; i<points.size(); i++)
    {
        delete points[i];
    }
    return 4;
}

int MEP::relaxSinglePoint_expensiveDecent(lua_State* L, int point, double h, int steps)
{
    // back up old sites so we can restore after
    vector<double> cfg;
    saveConfiguration(cfg);

    int good_steps = 0;
    int fail_steps = 0;
    
    const int num_sites = numberOfSites();
    const int ns3 = num_sites * 3;

    vector<VectorCS> vec;
    vector<VectorCS> vec2;
    vector<VectorCS> grad;

    getPoint(point, vec);

    vector<double> mags;
    for(int i=0; i<num_sites; i++)
    {
	mags.push_back( vec[i].magnitude() );
    }

    double vec_energy = vecEnergy(L, vec);

    computeVecFirstDerivative(L, vec, vec_energy, grad);
    for(int i=0; i<num_sites; i++)
	grad[i].zeroRadialComponent();

    double current_grad = sqrt(cart_norm2(grad));

    double* hh = new double[ns3];
    for(int i=0; i<ns3; i++)
        hh[i] = h;

#if 0
    for(int i=0; i<steps; i++)
    {
        copyVectorTo(vec, vec2);
	for(int qq=0; qq<ns3; qq++)
            vectorElementChange(vec2, qq, hh[qq], fixedRadius);

        rescale_vectors(vec2, mags);
        
        double vec2_energy = vecEnergy(L, vec2);
        
        computeVecFirstDerivative(L, vec2, vec2_energy, grad);
        for(int i=0; i<num_sites; i++)
            grad[i].zeroRadialComponent();
        const double new_grad = sqrt(cart_norm2(grad));
        
        if(new_grad <= current_grad)
        {
            copyVectorTo(vec2, vec);
            current_grad = new_grad;
            good_steps++;

            for(int qq=0; qq<ns3; qq++)
                hh[qq] *= 1.1;
        }
        else
        {
            int max_idx = 0;
            for(int qq=1; qq<ns3; qq++)
                if(fabs(hh[qq]) > fabs(hh[max_idx]))
                    max_idx = qq;

            hh[max_idx] *= -0.5;
        }
            
    }
#endif


#if 1
    for(int i=0; i<steps; i++)
    {
	for(int qq=0; qq<num_sites*3; qq++)
	{
	    copyVectorTo(vec, vec2);

	    if(vectorElementChange(vec2, qq, hh[qq], fixedRadius))
            {
                rescale_vectors(vec2, mags);

                double vec2_energy = vecEnergy(L, vec2);

                computeVecFirstDerivative(L, vec2, vec2_energy, grad);
                for(int i=0; i<num_sites; i++)
                    grad[i].zeroRadialComponent();
                const double new_grad = sqrt(cart_norm2(grad));

                if(new_grad <= current_grad)
                {
                    copyVectorTo(vec2, vec);
                    current_grad = new_grad;
                    good_steps++;
                    hh[qq] *= 1.1;
                }
                else
                    hh[qq] *= -0.2;
            }
        }
    }
#endif

    delete [] hh;

    setPoint(point, vec);

    energy_ok = false;

    loadConfiguration(cfg);

    return good_steps;
}

int MEP::expensiveEnergyMinimization(lua_State* L, int point, double h, int steps)
{
    // back up old sites so we can restore after
    vector<double> cfg;
    saveConfiguration(cfg);

    int good_steps = 0;
    const int ns = numberOfSites();

    vector<VectorCS> vec;
    vector<VectorCS> vec2;

    getPoint(point, vec);
    getPoint(point, vec2);

    setAllSpins(vec);
    double current_energy = getEnergy(L, vec);
    double start_energy = current_energy;

    bool* changer = new bool[ns*3];
    double* hi = new double[ns*3];
    copyVectorTo(vec, vec2);

    for(int i=0; i<ns*3; i++)
    {
        hi[i] = h;
        // pre-recording which sites change
        changer[i] = vectorElementChange(vec2, i, 0, fixedRadius);
    }

    for(int i=0; i<steps; i++)
    {
        int biggest_h = -1;
	for(int qq=0; qq<ns*3; qq++)
	{
            // don't consider unchanging
            if(changer[qq])
            {
                if(biggest_h < 0 || fabs(hi[qq]) > fabs(hi[biggest_h]))
                    biggest_h = qq;
            }
        }

        int qq = biggest_h;
        {
            bool change = false;
            copyVectorTo(vec, vec2);
            change = vectorElementChange(vec2, qq,  hi[qq], fixedRadius);

            if(change)
            {
                //printf("%2d  %4d/%4d\n", qq, i, steps );
                setAllSpins(vec2);
                double new_energy = getEnergy(L, vec2);
		
                if(new_energy < current_energy)
                {
                    //printf("                    %e\n", (current_energy-new_energy) );

                    current_energy = new_energy;
                    hi[qq] *= 2.1;
                    
                    copyVectorTo(vec2, vec);
                    good_steps++;
                }
                else
                {
                    hi[qq] *= -0.1; // notice we switch directions on failures
                }
            }
        }
    }

    delete [] hi;
    delete [] changer;

    setPoint(point, vec);
    energy_ok = false;
    loadConfiguration(cfg);

    lua_pushinteger(L, good_steps);
    lua_pushnumber(L, start_energy);
    lua_pushnumber(L, current_energy);
    lua_pushnumber(L, h);
    return 4;
}


int MEP::expensiveEnergyMaximization(lua_State* L, int point, double h, int steps)
{
    // back up old sites so we can restore after
    vector<double> cfg;
    saveConfiguration(cfg);

    int good_steps = 0;
    const int ns = numberOfSites();

    vector<VectorCS> vec;
    vector<VectorCS> vec2;

    getPoint(point, vec);
    getPoint(point, vec2);

    setAllSpins(vec);
    double current_energy = getEnergy(L, vec);
    double start_energy = current_energy;

    bool* changer = new bool[ns*3];
    double* hi = new double[ns*3];
    copyVectorTo(vec, vec2);

    for(int i=0; i<ns*3; i++)
    {
        hi[i] = h;
        // pre-recording which sites change
        changer[i] = vectorElementChange(vec2, i, 0, fixedRadius);
    }

    for(int i=0; i<steps; i++)
    {
        int biggest_h = -1;
	for(int qq=0; qq<ns*3; qq++)
	{
            // don't consider unchanging
            if(changer[qq])
            {
                if(biggest_h < 0 || fabs(hi[qq]) > fabs(hi[biggest_h]))
                    biggest_h = qq;
            }
        }

        int qq = biggest_h;
        {
            bool change = false;
            copyVectorTo(vec, vec2);
            change = vectorElementChange(vec2, qq,  hi[qq], fixedRadius);

            if(change)
            {
                //printf("%2d  %4d/%4d\n", qq, i, steps );
                setAllSpins(vec2);
                double new_energy = getEnergy(L, vec2);
		
                if(new_energy > current_energy)
                {
                    //printf("                    %e\n", (current_energy-new_energy) );

                    current_energy = new_energy;
                    hi[qq] *= 2.1;
                    
                    copyVectorTo(vec2, vec);
                    good_steps++;
                }
                else
                {
                    hi[qq] *= -0.1; // notice we switch directions on failures
                }
            }
        }
    }

    delete [] hi;
    delete [] changer;

    setPoint(point, vec);
    energy_ok = false;
    loadConfiguration(cfg);

    lua_pushinteger(L, good_steps);
    lua_pushnumber(L, start_energy);
    lua_pushnumber(L, current_energy);
    lua_pushnumber(L, h);
    return 4;
}



int MEP::expensiveGradientMinimization(lua_State* L, int point, double h, int steps)
{
    // back up old sites so we can restore after
    vector<double> cfg;
    saveConfiguration(cfg);

    int good_steps = 0;
    const int ns = numberOfSites();

    vector<VectorCS> vec;
    vector<VectorCS> vec2;
    vector<VectorCS> grad;

    getPoint(point, vec);
    getPoint(point, vec2);

    double vec_energy = vecEnergy(L, vec);
    computeVecFirstDerivative(L, vec, vec_energy, grad);

    double start_grad2 = cart_norm2(grad);
    double current_grad2 = start_grad2;
    
    for(int i=0; i<steps; i++)
    {
	bool improvement = false;
	for(int qq=0; qq<ns*3; qq++)
	{
	    for(int d=0; d<2; d++) //direction
	    {
                bool change = true;
		copyVectorTo(vec, vec2);
		if(d == 0)
		    change = vectorElementChange(vec2, qq, -h, fixedRadius);
		else
		    change = vectorElementChange(vec2, qq,  h, fixedRadius);

                if(change)
                {
                    double vec2_energy = vecEnergy(L, vec2);

                    computeVecFirstDerivative(L, vec2, vec2_energy, grad);
                    double new_grad2 = cart_norm2(grad);
                    
                    if(new_grad2 < current_grad2)
                    {
                        current_grad2  = new_grad2;
                        improvement = true;
                        
                        copyVectorTo(vec2, vec);
                    }
                }
            }
	}

	if(improvement)
	{
	    good_steps++;
	    h *= 2;
	}
	else
	{
	    h *= 0.5;
	}
    }



    setPoint(point, vec);

    energy_ok = false;

    lua_pushinteger(L, good_steps);
    lua_pushnumber(L, start_grad2);
    lua_pushnumber(L, current_grad2 );
    lua_pushnumber(L, h);
    loadConfiguration(cfg);
    return 4;
}




// relax individual point. This is used to refine a maximal point
// expected on the stack:
// at 1, mep
// at 2, point number
// at 3, optional start h
// returns min (mag of grad)
int MEP::relaxSinglePoint_SteepestDecent(lua_State* L)
{
    int consecutive_fails = 0;
    int consecutive_successes = 0;

    const double step_up = 2.0;
    const double step_down = 0.1;

    const int num_sites = numberOfSites();
    const int path_length = numberOfImages();

    const char* end_reason[4] = {"finished number of steps", "reached goal", "", ""};
    int end_reason_i = 0;

    const int p = lua_tointeger(L, 2) - 1;
    if(p < 0 || p >= path_length)
	return luaL_error(L, "Point index out of bounds");

    // need to save current configuration
    vector<double> cfg;
    saveConfiguration(cfg);

    vector<VectorCS> vec;
    vector<VectorCS> vec2;

    getPoint(p, vec);
    vector<double> mags;

    for(int i=0; i<num_sites; i++)
    {
	mags.push_back( vec[i].magnitude() );
    }

    // write path point as it currently stands
    setAllSpins(vec);

    vector<VectorCS> grad;
    double* _grad_grad = new double[num_sites*3];
    double h = epsilon;

    if(lua_isnumber(L, 3))
	h = lua_tonumber(L, 3);

    if(h < 1e-10)
	h = 1e-10;
    h = 1e-10;

    double vec_energy = vecEnergy(L, vec);

    computeVecFirstDerivative(L, vec, vec_energy, grad);
    for(int i=0; i<num_sites; i++)
	grad[i].zeroRadialComponent();

    double base_grad2 = cart_norm2(grad);

    for(int qq=0; qq<num_sites*3; qq++)
    {
	getPoint(p, vec2);
	bool change = vectorElementChange(vec2, qq, -h, fixedRadius);
        if(change)
        {
            double vec2_energy = vecEnergy(L, vec2);
            computeVecFirstDerivative(L, vec2, vec2_energy, grad);
            for(int i=0; i<num_sites; i++)
                grad[i].zeroRadialComponent(vec2[i]);
            double x_minus_h = sqrt(cart_norm2(grad));
            
            
            getPoint(p, vec2);
            vec2_energy = vecEnergy(L, vec2);
            vectorElementChange(vec2, qq,  h, fixedRadius);
            computeVecFirstDerivative(L, vec2, vec2_energy, grad);
            for(int i=0; i<num_sites; i++)
                grad[i].zeroRadialComponent(vec2[i]);
            double x_plus_h = sqrt(cart_norm2(grad));

            _grad_grad[qq] = (x_plus_h - x_minus_h) / (2.0 * h);
        }
        else
        {
            _grad_grad[qq] = 0;
        }
    }

    vector<VectorCS> grad_grad;
    for(int i=0; i<num_sites; i++)
    {
	grad_grad.push_back(  VectorCS(_grad_grad[i*3+0], _grad_grad[i*3+1], _grad_grad[i*3+2], vec2[i].cs) );
	grad_grad[i].zeroRadialComponent(vec[i]);
    }

    double gg2 = cart_norm2(grad_grad);

    double gg = sqrt(gg2);
    if(gg == 0)
	gg = 1.0;

    // making the overall length of the vector equal to 1
    // TODO: think ths over. Probably doesn't work for sites with heterogenous coordinates
    for(int i=0; i<num_sites; i++)
    {
	grad_grad[i].v[0] /= gg;
	grad_grad[i].v[1] /= gg;
	grad_grad[i].v[2] /= gg;
    }

#if 0
    printf("(%s:%i) grad_grad:\n", __FILE__, __LINE__);
    print_vecv(grad_grad);

    printf("(%s:%i) vec:\n", __FILE__, __LINE__);
    getPoint(p, vec);
    print_vecv(vec);
#endif


    double goal = 0;
    if(lua_isnumber(L, 4))
	goal = lua_tonumber(L, 4);


    int max_steps = 50;
    int good_steps = 0;
    double min2 = base_grad2;
    const double start_min2 = min2;

    const double len_gg = gg;//sqrt(cart_norm2(grad_grad));

    // printf(">>>>>>>>  h=%e  min2 > g2  (%e, %e)\n", h, min2, goal*goal);
    while(h > (epsilon * 1e-200) && max_steps && (min2 > goal*goal))
    {
	for(int i=0; i<num_sites; i++)
	{
	    vec2[i] = VectorCS::axpy(-h, grad_grad[i], vec[i]);
	    vec2[i].setMagnitude( mags[i] );
	}

        double vec2_energy = vecEnergy(L, vec2);
	computeVecFirstDerivative(L, vec2, vec2_energy, grad);
	for(int i=0; i<num_sites; i++)
	    grad[i].zeroRadialComponent();
	double min2_2 = cart_norm2(grad);

	if(min2_2 < min2)
	{
	    consecutive_fails = 0;
	    consecutive_successes++;

	    vec.clear();
	    for(int i=0; i<num_sites; i++)
		vec.push_back( vec2[i].copy() );

	    for(int i=0; i<consecutive_successes; i++)
		h = h * step_up;
	    good_steps++;
	    //printf("+ %e\n", h);
#if 0
	    printf("good step: %e -> %e      h = %e\n", min2, min2_2, h);
	    //for(int i=0; i<num_sites*3; i++)
	    //	printf("%g ", vxyz[i]);
	    //printf("\n");
#endif
	    min2 = min2_2;
	}
	else
	{
	    consecutive_fails++;
	    consecutive_successes = 0;

	    for(int i=0; i<consecutive_fails; i++)
		h = h * step_down;
#if 0
	    printf("- %e\n", h);
	    printf(" bad step: %e -> %e      h = %e\n", min2, min2_2, h);
#endif
	}
	max_steps--;
    }

    // write updated cfg
    // setAllSpins(L, set_index, vxyz);

    delete [] _grad_grad;

    //need to restore saved configuration to SpinSystem
    loadConfiguration(cfg);

    lua_pushnumber(L, sqrt(min2));
    lua_pushnumber(L, h);
    lua_pushinteger(L, good_steps);

    energy_ok = false;

    setPoint(p, vec);

    return 3;
}



// relax individual point. This is used to refine a maximal point
// expected on the stack:
// at 1, mep
// at 2, point number
// at 3, direction
// at 4, num steps
// at 5, step size
// at 6, tolerance
// returns new step size
int MEP::slidePoint(lua_State* L)
{
    const int num_sites = numberOfSites();
    const int path_length = numberOfImages();

    const double step_up = 1.1;
    const double step_down = 0.9;

    const int p = lua_tointeger(L, 2) - 1;
    if(p < 0 || p >= path_length)
	return luaL_error(L, "Point index out of bounds");

    double direction = lua_tonumber(L, 3);
    if(direction < 0)
	direction = -1;
    else
	direction =  1;

    int num_steps = lua_tointeger(L, 4);
    double h = lua_tonumber(L, 5);

    if(h == 0)
	h = 1e-10;

    // back up old sites so we can restore after
    vector<double> cfg;
    saveConfiguration(cfg);

    const int n = num_sites;
    vector<VectorCS> vec;
    getPoint(p, vec);
    // double* vec = &state_xyz_path[p*n];

    //double* slope = new double[n];
    vector<VectorCS> slope;
    vector<VectorCS> state;
    //double* state = new double[n];

    //double* mags = new double[num_sites];
    vector<double> mags;
    for(int i=0; i<num_sites; i++)
    {
	mags.push_back( vec[i].magnitude() );
	// mags[i] = _magnitude(currentSystem, &(vec[i*3]));
    }

	
    // write path point as it currently stands (at this point)
    setAllSpins(vec);
    double last_energy = getEnergy(L, vec);
    int consecutive_fails = 0;
    int consecutive_successes = 0;


    for(int i=0; i<num_steps; i++)
    {
        double vec_energy = vecEnergy(L, vec);
	computeVecFirstDerivative(L, vec, vec_energy, slope);

	for(int q=0; q<num_sites; q++)
	{
	    double sf[3];
	    vec[q].scaleFactors(sf);
	    // slope[q].scale(sf);
	}

	// gradient  direction
	// the following lines looks odd. The idea is to take the
	// list of numbers representing the gradient of all spins
	// and scale it down so the length, if it were treated as an
	// n-dimensional cartesian vector, is one.  
	//_normalizeTo(slope, slope, MEP::Cartesian, 1.0, n);
	double length = sqrt(cart_norm2(slope));
	if(length == 0)
	    length = 1.0;

	for(int j=0; j<n; j++)
	{
	    state.push_back( VectorCS::axpy(h * direction / length, slope[j], vec[j]) );
	    state.back().setMagnitude( mags[j] );
	    //state[j] = vec[j] + h * slope[j] * direction;
	}

	//rescale_vectors(currentSystem, state, mags, num_sites);
	setAllSpins(state);
	double current_energy = getEnergy(L, state);

	// printf("(%s:%i)  %g %e %e    %e\n", __FILE__, __LINE__, direction, current_energy, last_energy, h);

	if(direction < 0)
	{
	    if(current_energy < last_energy)
	    {
		// printf("(%s:%i)\n", __FILE__, __LINE__);
		consecutive_successes++;
		consecutive_fails = 0;
		for(int j=0; j<consecutive_successes; j++)
		    h *= step_up;
				
		// getVector(p, vec);
		setPoint(p, state);
		// memcpy(vec, state, sizeof(double)*n);
		last_energy = current_energy;
	    }
	    else
	    {
		// printf("(%s:%i)\n", __FILE__, __LINE__);
		consecutive_successes=0;
		consecutive_fails++;
		for(int j=0; j<consecutive_fails; j++)
		    h *= step_down;
	    }
	}
	else
	{
	    if(current_energy > last_energy)
	    {
		// printf("(%s:%i)\n", __FILE__, __LINE__);
		consecutive_successes++;
		consecutive_fails = 0;
		for(int j=0; j<consecutive_successes; j++)
		    h *= step_up;
		setPoint(p, state);
		// memcpy(vec, state, sizeof(double)*n);
		last_energy = current_energy;
	    }
	    else
	    {
		// printf("(%s:%i)\n", __FILE__, __LINE__);
		consecutive_successes=0;
		consecutive_fails++;
		for(int j=0; j<consecutive_fails; j++)
		    h *= step_down;
	    }
	}

    }



    loadConfiguration(cfg);	
    energy_ok = false;
    // delete [] slope;
    // delete [] mags;
    // delete [] state;

    lua_pushnumber(L, last_energy);
    return 1;
}



int MEP::numberOfImages()
{
    if(sites.size() == 0)
	return 0;
    return state_path.size() / numberOfSites();
}

int MEP::numberOfSites()
{
    return sites.size() / 3;
}


int MEP::l_angle_between_pointsite(lua_State* L, int base)
{
    int p1 = lua_tointeger(L, base)-1;
    int s1 = lua_tointeger(L, base+1)-1;
    int p2 = lua_tointeger(L, base+2)-1;
    int s2 = lua_tointeger(L, base+3)-1;

    lua_pushnumber(L, VectorCS::angleBetween( getPointSite(p1,s1), getPointSite(p2,s2) ));
    return 1;
}

bool MEP::equalByAngle(int a, int b, double allowable)
{
    /*
      const int chunk = sites.size();
      double* s1 = &state_xyz_path[a * chunk];
      double* s2 = &state_xyz_path[b * chunk];
    */
    
    vector<VectorCS> s1;
    vector<VectorCS> s2;
    
    getPoint(a, s1);
    getPoint(b, s2);
    
    for(int i=0; i<numberOfSites(); i++)
    {
	if( VectorCS::angleBetween(s1[i], s2[i]) > allowable )
	    //if(_angleBetween(&s1[i*3], &s2[i*3], currentSystem, currentSystem) > allowable)
	    return false;
    }
    
    return true;
}



#include <set>
// sites are actually images/points
int MEP::uniqueSites(lua_State* L)
{
    const int ni = numberOfImages();

    double tol = 1e-4;
    vector<int> consider_sites;
    vector<int> unique_sites;
    set<int> sites_set;

    for(int i=2; i<=lua_gettop(L); i++)
    {
	if(lua_isnumber(L, i))
	{
	    tol = lua_tonumber(L, i);
	}

	if(lua_istable(L, i))
	{
	    lua_pushnil(L);
	    while(lua_next(L, i))
	    {
		if(lua_isnumber(L, -1))
		{
		    int a = lua_tointeger(L, -1) - 1;
		    if(a >= 0 && a < ni)
		    {
			sites_set.insert(a);
		    }
		}
		lua_pop(L, 1);
	    }
	}
    }

    if(sites_set.size() == 0)
    {
	for(int i=0; i<ni; i++)
	{
	    consider_sites.push_back(i);
	}
    }
    else
    {
	set<int>::iterator it;
	for(it=sites_set.begin(); it!=sites_set.end(); ++it)
	{
	    consider_sites.push_back(*it);
	}
    }

    for(unsigned int i=0; i<consider_sites.size(); i++)
    {
	bool is_unique = true;

	for(unsigned int j=0; is_unique &&  j<unique_sites.size(); j++)
	{
	    if(equalByAngle( consider_sites[i], unique_sites[j], tol))
	    {
		is_unique = false;				
	    }
	}

	if(is_unique)
	    unique_sites.push_back(consider_sites[i]);
    }

    lua_newtable(L);
    for(unsigned int i=0; i<unique_sites.size(); i++)
    {
	lua_pushinteger(L, i+1);
	lua_pushinteger(L, unique_sites[i]+1);
	lua_settable(L, -3);
    }

    return 1;
}

// unused
void MEP::computePointGradAtSite(lua_State* L, int p, int s, double* grad3)
{
#if 0
//	vector<double> cfg;
//  saveConfiguration(L, get_index, cfg);

    VectorCS vec;
    getPointSite(p, s, vec);

    const double m1 = vec.magnitude(); //_magnitude(currentSystem, vec);
	
    double h[3];
    vec.stepSize(epsilon, h);
    // _stepSize(currentSystem, vec, epsilon, h);

    // We now have a point at site p,s and appropriate step sizes
    // time to compute the spatial energy gradient.
    double g[6] = {0,0,0,0,0,0};

    // 7th below is to restore old state
    const double cc[7][3] = {{1,0,0},{-1,0,0},   {0,1,0},{0,-1,0},   {0,0,1},{0,0,-1},   {0,0,0}};
    for(int c=0; c<3; c++)
    {
	double f[4];
	const double d[5] = {-2,-1,1,2,0};
		
	for(int i=0; i<5; i++)
	{
	    VectorCS new_coord(vec);
	    new_coord.v[c] += d[i] * h[c];
	    new_coord.setMagnitude( m1 );
	    setSiteSpin(&sites[s*3], new_coord);

	    if(i<4)
	    {
		f[i] = getEnergy(L);
	    }
	}
	grad3[c] = (f[0]-8.0*f[1]+8.0*f[2]-f[3]) / (12.0 * h[c]);
    }

    double sf[3];
    vec.scaleFactors(sf);

    grad3[0] *= sf[0];
    grad3[1] *= sf[1];
    grad3[2] *= sf[2];


//	loadConfiguration(L, set_index, cfg);
#endif
}


static int addToTable(lua_State* L, int tab_pos, int index, int value)
{
    index++;
    lua_pushinteger(L, index);
    lua_pushinteger(L, value);
    lua_settable(L, tab_pos);
    return index;
}

int MEP::anglesBetweenPoints(lua_State* L)
{
    const int ni = numberOfImages();
    const int ns = numberOfSites();

    if(!lua_isnumber(L, 2) || !lua_isnumber(L, 3))
	return luaL_error(L, "Require 2 numbers");

    int a = lua_tointeger(L, 2) - 1;
    int b = lua_tointeger(L, 3) - 1;
	
    if( (a < 0) || (b < 0) || (a >= ni) || (b >= ni))
	return luaL_error(L, "Require 2 numbers between 1 and number of points (%i)", ni);

    vector<VectorCS> va;
    vector<VectorCS> vb;

    getPoint(a, va);
    getPoint(b, vb);
	   
    lua_newtable(L);
    for(int i=0; i<ns; i++)
    {
	lua_pushinteger(L, i+1);

	double angle = VectorCS::angleBetween(va[i], vb[i]);
	lua_pushnumber(L, angle);

	lua_settable(L, -3);
    }
		

    return 1;
}


int MEP::maxpoints(lua_State* L)
{
    const int path_length = numberOfImages();
	
    lua_newtable(L); //mins
    const int min_idx = lua_gettop(L);

    lua_newtable(L); //maxs
    const int max_idx = lua_gettop(L);
	
    lua_newtable(L); //all
    const int all_idx = lua_gettop(L);

    if(energies.size() != path_length || energies.size() == 0)
	return luaL_error(L, "Energy path size mismatch, expected %d and greater than zero, got %d", path_length,  energies.size());

    int szMin = 0;
    int szMax = 0;
    int szAll = 0;
	
    if(energies[1] > energies[0]) // start at min
    {
	szMin = addToTable(L, min_idx, szMin, 1);
	szAll = addToTable(L, all_idx, szAll, 1);
    }
    else
    {
	szMax = addToTable(L, max_idx, szMax, 1);
	szAll = addToTable(L, all_idx, szAll, 1);
    }
	
    for(int i=1; i<path_length-1; i++)
    {
	const double a = energies[i-1];
	const double b = energies[i];
	const double c = energies[i+1];
		
	if(b<=a && b<c) //local minimum
	{
	    szMin = addToTable(L, min_idx, szMin, i+1);
	    szAll = addToTable(L, all_idx, szAll, i+1);
	}

	if(b>a && b>=c) //local maximum
	{
	    szMax = addToTable(L, max_idx, szMax, i+1);
	    szAll = addToTable(L, all_idx, szAll, i+1);
	}
    }

    if(energies[path_length-2] > energies[path_length-1]) // end at min
	szMin = addToTable(L, min_idx, szMin, path_length);
    else
	szMax = addToTable(L, max_idx, szMax, path_length);

    szAll = addToTable(L, all_idx, szAll, path_length);

    return 3;
}

int MEP::calculateEnergyGradients(lua_State* L)
{
    const int ns = numberOfSites();
    const int ni = numberOfImages();
	
    force_vector.resize( state_path.size() ); //since we're doing random-ish access

    double sf[3] = {1,1,1};
	
    // lets march along the path
    for(int i=0; i<ni; i++)
    {
	vector<VectorCS> vec;
	vector<VectorCS> force;
	getPoint(i, vec);
	getForcePoint(i, force);

        double vec_energy = vecEnergy(L, vec);
	computeVecFirstDerivative(L, vec, vec_energy, force);

	for(int j=0; j<ns; j++)
	{
	    force[j].zeroRadialComponent(vec[j]);
	}

	/*
	  printf("force %i\n", i);
	  print_vecv(force);
	*/

	setForcePoint(i, force);
    }

    return 0;
}

// get energies about a state
void MEP::stateNeighbourhood(const vector<VectorCS>& _state, lua_State* L, const vector<double>& h, vector<double>& eee_h)
{
    const int ns = numberOfSites();
    const int np = numberOfImages();

    // back up old sites so we can restore after
    vector<double> cfg;
    saveConfiguration(cfg);

    vector<double> mags;
    for(int i=0; i<ns; i++)
	mags.push_back( _state[i].magnitude() );

    setAllSpins(_state);
    const double base_energy = getEnergy(L, _state);
    double e1, e2;

    for(int i=0; i<ns*3; i++)
    {
	bool change;
	{
	    vector<VectorCS> state(_state);
	    change = vectorElementChange(state, i, -h[i], fixedRadius);
	    rescale_vectors(state, mags);
	    setAllSpins(state);
	    e1 = getEnergy(L, state);
	}

	{
	    vector<VectorCS> state(_state);
	    vectorElementChange(state, i,  h[i], fixedRadius);
	    rescale_vectors(state, mags);
	    setAllSpins(state);
	    e2 = getEnergy(L, state);
	}

	if(change)
	{
	    eee_h.push_back(e1);
	    eee_h.push_back(base_energy);
	    eee_h.push_back(e2);
	    eee_h.push_back(h[i]);
	}
    }

    loadConfiguration(cfg);
}

// get energies about a point
void MEP::pointNeighbourhood(const int point, lua_State* L, const vector<double>& h, vector<double>& eee_h)
{
    vector<VectorCS> state;
    getPoint(point, state);

    stateNeighbourhood(state, L, h, eee_h);
}

int MEP::l_classifyPoint(lua_State* L, int base)
{
    int point = lua_tointeger(L, base)-1;
    // int type = lua_tointeger(L, base+1);

    vector<double> h;
    for(int i=0; i<numberOfSites()*3; i++)
    {
	lua_pushinteger(L, i+1);
	lua_gettable(L, base+1);
	h.push_back( lua_tonumber(L, -1));
	lua_pop(L, 1);
    }

    vector<double> eee_h;

    pointNeighbourhood(point, L, h, eee_h);

    for(int i=0; i<eee_h.size(); i+=4)
    {
	printf("%e %e %e %e\n", eee_h[i*4+0], eee_h[i*4+1], eee_h[i*4+2], eee_h[i*4+3]);
    }

    for(int type=-1; type<2; type++)
    {
	lua_pushnumber(L, classifyNeighbourhood(type, eee_h));
    }
    return 3;
}


// type: -1=min, 0=saddle, 1=max
// The plan here is to quantify how much an energy neighbourhood resembles these features
double MEP::classifyNeighbourhood(int type, const vector<double>& eee_h)
{
    vector<double> c2;
    double sum = 0;
    for(unsigned int i=0; i<eee_h.size(); i+=4)
    {
	const double a = eee_h[i*4+0];
	const double b = eee_h[i*4+1];
	const double c = eee_h[i*4+2];
	const double x = eee_h[i*4+3];

	if(x == 0)
	    c2.push_back(0);
	else
	{
	    const double v = (c - 2.0 * b + a) / (x*x);
	    c2.push_back(v);
	    sum += v;
	}
    }
    
    if(type == -1)
    {
	return sum;
    }
    if(type == 1)
    {
	return -sum;
    }

    // for a saddle point we want one negative curvature and the rest positive
    // so we'll negate our most negative curvature and return the sum
    int most_negative_index = 0;
    for(unsigned int i=1; i<c2.size(); i++)
    {
	if(c2[i] < c2[most_negative_index])
	    most_negative_index = i;
    }

    c2[most_negative_index] = -c2[most_negative_index];

    sum = 0;
    for(unsigned int i=0; i<c2.size(); i++)
    {
	sum += c2[i];
    }
    return sum;
}



int MEP::classifyPoint(lua_State* L)
{
    const int num_sites = numberOfSites();
    const int path_length = numberOfImages();

    const int p = lua_tointeger(L, 2) - 1;
    if(p < 0 || p >= path_length)
	return luaL_error(L, "Point index out of bounds");

    vector<double> h;

    lua_pushnil(L);
    while(lua_next(L, 3))
    {
	h.push_back(lua_tonumber(L, -1));
	lua_pop(L, 1);
    }

    if(h.size() != num_sites * 3)
	return luaL_error(L, "wrong number of step sizes");

    // back up old sites so we can restore after
    vector<double> cfg;
    saveConfiguration(cfg);

    vector<VectorCS> state;
    getPoint(p, state);

    vector<double> mags;

    for(int i=0; i<num_sites; i++)
    {
	mags.push_back( state[i].magnitude() ); //_magnitude(currentSystem, &(vec[i*3]));
    }

    setAllSpins(state);
    const double base_energy = getEnergy(L, state);

    int ups = 0;
    int downs = 0;
    int equals = 0;
	
    lua_newtable(L);
    for(int i=0; i<num_sites*3; i++)
    {
	lua_pushinteger(L, i+1);
	lua_newtable(L);

	getPoint(p, state);
	vectorElementChange(state, i, -h[i], fixedRadius);
	rescale_vectors(state, mags);
	setAllSpins(state);
	const double e1 = getEnergy(L, state);

	getPoint(p, state);
	vectorElementChange(state, i,  h[i], fixedRadius);
	rescale_vectors(state, mags);
	setAllSpins(state);
	const double e2 = getEnergy(L, state);

	lua_pushinteger(L, 1);
	lua_pushnumber(L, e1);
	lua_settable(L, -3);

	lua_pushinteger(L, 2);
	lua_pushnumber(L, base_energy);
	lua_settable(L, -3);

	lua_pushinteger(L, 3);
	lua_pushnumber(L, e2);
	lua_settable(L, -3);


	lua_settable(L, -3);
    }
    return 1;
}


static int l_applyforces(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->applyForces(L);	
}


static int l_setdata(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    luaL_unref(L, LUA_REGISTRYINDEX, mep->ref_data);
    lua_pushvalue(L, 2);
    mep->ref_data = luaL_ref(L, LUA_REGISTRYINDEX);
    return 0;
}

static int l_getdata(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    lua_rawgeti(L, LUA_REGISTRYINDEX, mep->ref_data);
    return 1;
}

static int l_setrfm(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    mep->relax_direction_fail_max = lua_tointeger(L, 2);
    return 0;
}
static int l_getrfm(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    lua_pushinteger(L, mep->relax_direction_fail_max);
    return 1;
}

static int l_addsite(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
	
    int site[3] = {0,0,0};
	
    for(int i=0; i<3; i++)
    {
	if(lua_isnumber(L, 2+i))
	{
	    site[i] = lua_tointeger(L, 2+i) - 1;
	}
    }
	
    mep->addSite(site[0], site[1], site[2]);
    return 0;
}

static int l_clearsites(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    mep->sites.clear();
    return 0;
}
static int l_clearpath(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    mep->state_path.clear();
    return 0;
}

static int l_getallsites(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    lua_newtable(L);
	
    for(unsigned int i=0; i<mep->sites.size()/3; i++)
    {
	lua_pushinteger(L, i+1);
	lua_newtable(L);
	for(int j=0; j<3; j++)
	{
	    lua_pushinteger(L, j+1);
	    lua_pushinteger(L, mep->sites[i*3+j]+1);
	    lua_settable(L, -3);
	}
	lua_settable(L, -3);
    } 	
    return 1;
}


static int l_addstatexyz(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    VectorCS vv = lua_toVectorCS(L, 2);
    mep->state_path.push_back( vv );
    mep->energy_ok = false;

    return 0;
}

static int l_setimagesitemobility(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
	
    const int i = lua_tointeger(L, 2)-1;
    const int s = lua_tointeger(L, 3)-1;
    const double m = lua_tonumber(L, 4);
	
    mep->setImageSiteMobility(i, s, m);
	
    return 0;
}

static int l_getimagesitemobility(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
	
    const int i = lua_tointeger(L, 2)-1;
    const int s = lua_tointeger(L, 3)-1;

    lua_pushnumber(L, mep->getImageSiteMobility(i,s));
	
    return 1;
}


static int l_resampleStateXYZPath(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->resampleStatePath(L);
}

static int l_projForcePerpSpins(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
	
    mep->projForcePerpSpins(L);
    return 0;
}
static int l_projForcePerpPath(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    mep->projForcePerpPath(L);
    return 0;
}
static int l_projForcePath(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
	
    mep->projForcePath(L);
    return 0;
}

static int l_calculateEnergyGradients(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
	
    return mep->calculateEnergyGradients(L);
}

static int l_calculateEnergies(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
	
    return mep->calculateEnergies(L);
}

static int l_calculateSingleEnergy(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
	
    return mep->calculateSingleEnergy(L);
}



static int l_getpathenergy(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    lua_newtable(L);
    for(int i=0; i<mep->energies.size(); i++)
    {
	lua_pushinteger(L, i+1);
	lua_pushnumber(L, mep->energies[i]);
	lua_settable(L, -3);
    }
    return 1;
}

static int l_getgradient(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    // positive values count from beginning,
    // negative values count from end
    int p = lua_tointeger(L, 2);
    int s = lua_tointeger(L, 3);

    if(p < 0)
    {
	p = mep->numberOfImages() + p;
    }
    if(s < 0)
    {
	s = mep->numberOfSites() + s;
    }

    p--;
    s--;

    if(!mep->validPointSite(p,s))
    {
	return luaL_error(L, "Invalid point or site");
    }

    int idx = mep->numberOfSites() * p + s;

    if(idx >= mep->force_vector.size())
	return 0;

    return lua_pushVectorCS(L, mep->force_vector[idx], VCSF_CSDESC);
}

static int l_relaxSinglePoint_sd_(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    return mep->relaxSinglePoint_SteepestDecent(L);
}


static int l_relaxSinglePoint_ex_(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    int point = lua_tointeger(L, 2) - 1;
    if(point <0 || point >= mep->numberOfImages())
        return luaL_error(L, "Invalid point index");

    double h = lua_tonumber(L, 3);

    int steps = lua_tointeger(L, 4);

    int n = mep->relaxSinglePoint_expensiveDecent(L, point, h, steps);

    lua_pushnumber(L, n);
    return 1;
}


static int l_slidePoint_(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->slidePoint(L);
}


// expected on the stack at function call
// at 1, mep
// at 2, point number
// this is the internal version
static int l_computepoint2deriv(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    int num_sites = mep->numberOfSites();
	
    int p = lua_tointeger(L, 2) - 1;
	
    if(p < 0 || p >= mep->numberOfImages())
	return luaL_error(L, "Invalid point number");

    double* derivsAB = new double[num_sites * num_sites * 9];
	
    mep->computePointSecondDerivative(L, p, derivsAB);
	
    lua_newtable(L);
    for(int i=0; i<num_sites * num_sites * 9; i++)
    {
	lua_pushinteger(L, i+1);
	lua_pushnumber(L, derivsAB[i]);
	lua_settable(L, -3);
    }
    delete [] derivsAB;
    return 1;
}	
	

// expected on the stack at function call
// at 1, mep
// at 2, point number
// this is the internal version
static int l_computepoint1deriv(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    int num_sites = mep->numberOfSites();
	
    int p = lua_tointeger(L, 2) - 1;
	
    if(p < 0 || p >= mep->numberOfImages())
	return luaL_error(L, "Invalid point number");

    //double* derivs = new double[num_sites * 3];
	
    vector<VectorCS> derivs;

    vector<VectorCS> vec;
    mep->getPoint(p, vec);
    double p_energy = mep->vecEnergy(L, vec);


    mep->computePointFirstDerivative(L, p, p_energy, derivs);

    lua_newtable(L);
    int j = 1;
    for(int i=0; i<derivs.size(); i++)
    {
	for(int q=0; q<3; q++)
	{
	    lua_pushinteger(L, j);
	    lua_pushnumber(L, derivs[i].v[q]);
	    lua_settable(L, -3);
	    j++;
	}
    }
    return 1;
}	
	

static int l_getsite(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    // positive values count from beginning,
    // negative values count from end
    int p = lua_tointeger(L, 2);
    int s = lua_tointeger(L, 3);

    if(p < 0)
    {
	p = mep->numberOfImages() + p;
    }
    if(s < 0)
    {
	s = mep->numberOfSites() + s;
    }

    p--;
    s--;

    if(!mep->validPointSite(p,s))
    {
	return luaL_error(L, "Path Point or site is out of bounds. {Point,Site} = {%d,%d}. Upper Bound = {%d,%d}", p+1,s+1,mep->numberOfImages(), mep->numberOfSites());
    }

    VectorCS cc = mep->getPointSite(p,s).convertedToCoordinateSystem(Cartesian);

    for(int i=0; i<3; i++)
	lua_pushnumber(L, cc.v[i]);

    lua_pushnumber(L, cc.magnitude());
    return 4;
	
    //return lua_pushVectorCS(L, mep->getPointSite(p,s).convertedToCoordinateSystem(Cartesian), VCSF_CSDESC);
}


static int l_nativespin(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    // positive values count from beginning,
    // negative values count from end
    int p = lua_tointeger(L, 2);
    int s = lua_tointeger(L, 3);

    if(p < 0)
    {
	p = mep->numberOfImages() + p;
    }
    if(s < 0)
    {
	s = mep->numberOfSites() + s;
    }

    p--;
    s--;

    if(!mep->validPointSite(p,s))
    {
	return luaL_error(L, "Path Point or site is out of bounds. {Point,Site} = {%d,%d}. Upper Bound = {%d,%d}", p+1,s+1,mep->numberOfImages(), mep->numberOfSites());
    }

    return lua_pushVectorCS(L, mep->getPointSite(p,s), VCSF_CSDESC);
}

	
static int l_setsite(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    // positive values count from beginning,
    // negative values count from end
    int p = lua_tointeger(L, 2);
    int s = lua_tointeger(L, 3);

    if(p < 0)
    {
	p = mep->numberOfImages() + p;
    }
    if(s < 0)
    {
	s = mep->numberOfSites() + s;
    }

    p--;
    s--;

    if(!mep->validPointSite(p,s))
    {
	return luaL_error(L, "Path Point or site is out of bounds. {Point,Site} = {%d,%d}. Upper Bound = {%d,%d}", p+1,s+1,mep->numberOfImages(), mep->numberOfSites());
    }

    VectorCS v = lua_toVectorCS(L, 4);

    mep->setPointSite(p,s, v.convertedToCoordinateSystem(mep->getCSAt(p,s)));
	
    mep->energy_ok = false;
    mep->good_distances = false;
	
    return 0;
}


static int l_sitecount(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    lua_pushinteger(L, mep->numberOfSites());
	
    return 1;
}
static int l_randomize_(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    mep->randomize(lua_tonumber(L, 2));
    return 0;
}

static int l_setbeta(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    mep->beta = lua_tonumber(L, 2);
    return 0;
}

static int l_getbeta(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    lua_pushnumber(L, mep->beta);
    return 1;
}

static int l_internal_copyto(lua_State* L)
{
    LUA_PREAMBLE(MEP, src, 1);
    LUA_PREAMBLE(MEP, dest, 2);

    src->internal_copy_to(dest);
    return 0;
}

static int l_maxpoints(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    return mep->maxpoints(L);	
}

static int l_absoluteDifference(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep1, 1);
    LUA_PREAMBLE(MEP, mep2, 2);

    int point = -1;
    if(lua_isnumber(L, 3))
    {
	point = lua_tointeger(L, 3) - 1; // c->lua base change
    }

    double max_diff;
    int max_idx;
    vector<double> vdiff;
    const double d = mep1->absoluteDifference(mep2, point, max_diff, max_idx, vdiff);

    if(d == -1)
	return luaL_error(L, "internal state size mismatch");

    lua_pushnumber(L, d);
    lua_newtable(L);
    for(int i=0; i<vdiff.size(); i++)
    {
        lua_pushinteger(L, i+1);
        lua_pushnumber(L, vdiff[i]);
        lua_settable(L, -3);
    }
    //lua_pushnumber(L, max_diff);
    //lua_pushinteger(L, max_idx+1);
    return 2;
}

static int l_ppc(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    lua_pushinteger(L, mep->numberOfImages());
    return 1;
}

static int l_pend_(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
	
    int n = 0;
    if(lua_isnumber(L, 2))
	n = lua_tonumber(L, 2);

    vector<double> nderiv;
	
    mep->calculatePathEnergyNDeriv(L, n, nderiv);
	
    double total = mep->d12_total;
	
    lua_newtable(L); //energy
    lua_newtable(L); //location
	
    vector<double> normalized_distance;
    normalized_distance.push_back(0);
    for(int i=0; i<mep->d12.size(); i++)
    {
	normalized_distance.push_back(normalized_distance[i] + mep->d12[i] / mep->d12_total);
    }
	
    for(int i=0; i<nderiv.size(); i++)
    {
	const double ederiv_here = nderiv[i];
	const double pos_here = normalized_distance[i];
		
	lua_pushinteger(L, i+1);
	lua_pushnumber(L, ederiv_here);
	lua_settable(L, -4);
		
	lua_pushinteger(L, i+1);
	lua_pushnumber(L, pos_here);
	lua_settable(L, -3);
    }
		
    lua_newtable(L);
    int j = 0;
    for(int i=1; i<nderiv.size(); i++)
    {
	if( (nderiv[i] * nderiv[i-1]) < 0) //sign change
	{
	    const double v0 = nderiv[i-1];
	    const double v1 = nderiv[i];
			
	    const double p0 = normalized_distance[i-1];
	    const double p1 = normalized_distance[i];
			
	    const double t = -v0 * ((p1-p0) / (v1-v0));
			
	    const double pCrossing = p0 + t * (p1-p0);
			
	    j++;
	    lua_pushinteger(L, j);
	    lua_pushnumber(L, pCrossing);
	    lua_settable(L, -3);
	}
    }

    return 3;
}


static int _l_getCoordinateSystems(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->l_getCoordinateSystems(L);
}

static int _l_setCoordinateSystem(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->l_setCoordinateSystem(L,2);
}

static int _l_getCoordinateSystem(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->l_getCoordinateSystem(L);
}

static int l_getep(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    lua_pushnumber(L, mep->epsilon);
    return 1;
}

static int l_setep(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    mep->epsilon = lua_tonumber(L, 2);
    return 0;
}

static int l_asbs(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->anglesBetweenPoints(L);
}

static int l_us(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->uniqueSites(L);
}

static int l_cp_(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->classifyPoint(L);
}

static void pq(VectorCS& v)
{
    printf("% 4e  % 4e  % 4e %s\n", v.v[0], v.v[1], v.v[2], nameOfCoordinateSystem(v.cs));
}
static int l_conv_cs(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    int consume;

    VectorCS v = lua_toVectorCS(L, 2, consume);
    //print_vec(v);

    const char* dest_cs = lua_tostring(L, 2+consume);
    //printf("c = %i, dest = %s\n", consume, dest_cs);
    v.convertToCoordinateSystem( coordinateSystemByName(dest_cs) );
    //print_vec(v);

    return lua_pushVectorCS(L, v, VCSF_CSDESC);
}


static int _l_angle_between_pointsite(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->l_angle_between_pointsite(L,2);
}

static int _l_expensiveenergyminimization(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    int point = lua_tointeger(L, 2)-1;
    double h = lua_tonumber(L, 3);
    int steps = lua_tointeger(L, 4);

    return mep->expensiveEnergyMinimization(L, point, h, steps);
}
static int _l_expensiveenergymaximization(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    int point = lua_tointeger(L, 2)-1;
    double h = lua_tonumber(L, 3);
    int steps = lua_tointeger(L, 4);

    return mep->expensiveEnergyMaximization(L, point, h, steps);
}
static int _l_expensivegradientminimization(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    int point = lua_tointeger(L, 2)-1;
    double h = lua_tonumber(L, 3);
    int steps = lua_tointeger(L, 4);

    return mep->expensiveGradientMinimization(L, point, h, steps);
}

static int l_computePointSecondDerivativeAB(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    const int pt = lua_tonumber(L, 2) - 1;
    const int c1 = lua_tonumber(L, 3) - 1;
    const int c2 = lua_tonumber(L, 4) - 1;

    double d1 = -1;
    double d2 = -1;

    if(lua_isnumber(L, 5))
	d1 = lua_tonumber(L, 5);
    if(lua_isnumber(L, 6))
	d2 = lua_tonumber(L, 6);

    lua_pushnumber(L, mep->computePointSecondDerivativeAB(L, pt, c1, c2, d1, d2));

    return 1;
}

static int _l_resize(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->_resize(L, 2);
}

static int _l_swap(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->_swap(L, 2);
}

static int _l_copy(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->_copy(L, 2);
}

static int _l_rpab(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->l_rotatePathAboutBy(L, 2);
}

static int _l_classifypoint(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->l_classifyPoint(L, 2);
}

static int _l_setspinsystem(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->lua_setspinsystem(L);
}

static int _l_getspinsystem(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->lua_getspinsystem(L);
}

static int _l_setEnergyFunction(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->lua_setEnergyFunction(L);
}

static int _l_getEnergyFunction(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->lua_getEnergyFunction(L);
}

static int _l_findlowgradislands(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);
    return mep->findCriticalIslands(L);
}

static int _l_int_cp(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

    vector<VectorCS> v1;
    vector<VectorCS> v2;
    vector<VectorCS> v3;

    lua_pushnil(L);
    while(lua_next(L, 2))
    {
	v1.push_back( lua_toVectorCS(L, lua_gettop(L) ) );
	lua_pop(L, 1);
    }

    lua_pushnil(L);
    while(lua_next(L, 3))
    {
	v2.push_back( lua_toVectorCS(L, lua_gettop(L) ) );
	lua_pop(L, 1);
    }

    double ratio = lua_tonumber(L, 4);

    bool backwards = false;
    if(lua_isboolean(L, 5))
        backwards = lua_toboolean(L, 5);

    mep->interpolateVectorCS(v1, v2, ratio, v3, backwards);


    lua_newtable(L);
    for(int i=0; i<v3.size(); i++)
    {
	lua_pushinteger(L, i+1);
	lua_pushVectorCS(L, v3[i], VCSF_ASTABLE | VCSF_CSDESC);
	lua_settable(L, -3);
    }
    return 1;
}



#include "mep_bestpath.h"
static int _l_bestpath(lua_State* L)
{
    return l_bestpath(L);
}


int MEP::help(lua_State* L)
{
#if 0
    Moving this chunk to the lua file
	if(lua_gettop(L) == 0)
	{
	    lua_pushstring(L, "Calculates a minimum energy pathway between two states.");
	    //lua_pushstring(L, "1 *3Vector* or *SpinSystem*: System Size");
	    lua_pushstring(L, "");
	    lua_pushstring(L, ""); //output, empty
	    return 3;
	}

    if(lua_istable(L, 1))
    {
	return 0;
    }
#endif

    lua_CFunction func = lua_tocfunction(L, 1);

    if(func == l_setdata)
    {
	lua_pushstring(L, "Set internal data. This method is used in the support lua scripts and should not be used.");
	lua_pushstring(L, "1 Value: the new internal data");
	lua_pushstring(L, "");
	return 3;
    }
    if(func == l_getdata)
    {
	lua_pushstring(L, "Get internal data. This method is used in the support lua scripts and should not be used.");
	lua_pushstring(L, "");
	lua_pushstring(L, "1 Value: the internal data");
	return 3;
    }
    if(func == &(l_getsite))
    {
	lua_pushstring(L, "Get site as Cartesian Coordinates");
	lua_pushstring(L, "2 Integers: 1st integer is path index, 2nd integer is site index. Positive values count from the start, negative values count from the end.");
	lua_pushstring(L, "4 Numbers: Cartesian coordinates and magnitude at point p and site s.");
	return 3;
    }
    if(func == &(l_setsite))
    {
	lua_pushstring(L, "Set site direction and magnitude using Cartesian Coordinates");
	lua_pushstring(L, "2 Integers, 3 Numbers or 1 table of 3 Numbers: 1st integer is path index, 2nd integer is site index, 3 numbers are the new x,y and z coordinates.");
	lua_pushstring(L, "");
	return 3;
    }
	
    if(func == l_getallsites)
    {
	lua_pushstring(L, "Get which sites are involved in calculation");
	lua_pushstring(L, "");
	lua_pushstring(L, "1 Table of Tables of 3 Integers: Site positions involved in calculation.");
	return 3;
    }
	
    if(func == l_sitecount)
    {
	lua_pushstring(L, "Get number of sites invovled in the calculation");
	lua_pushstring(L, "");
	lua_pushstring(L, "1 Integer: number of sites involved in the calculation.");
	return 3;
    }
    /*
      if(func == l_ppc)
      {
      lua_pushstring(L, "Get number of path points that currently exist in the object.");
      lua_pushstring(L, "");
      lua_pushstring(L, "1 Integer: number of path points.");
      return 3;
      }
    */
    if(func == l_absoluteDifference)
    {
	lua_pushstring(L, "Get the absolute difference between the calling MEP and the passed MEP. Difference between two moments m1 and m2 is defined as the sum of all pairwise angular differences.");
	lua_pushstring(L, "1 MEP: MEP to compare against.");
	lua_pushstring(L, "1 Number, 1 Table fo Numbers: Total of all differences, the difference at each site.");
	return 3;
    }

    if(func == l_setbeta)
    {
	lua_pushstring(L, "Set the internal scaling factor that converts between energy gradient and displacement. This value gets adapted automatically. If a close value for Beta is knows at the start of the calculation it will save a few iterations if it is set explicitly.");
	lua_pushstring(L, "1 Number: Beta");
	lua_pushstring(L, "");
	return 3;
    }
    if(func == l_getbeta)
    {
	lua_pushstring(L, "Get the internal scaling factor that converts between energy gradient and displacement. This value gets adapted automatically.");
	lua_pushstring(L, "");
	lua_pushstring(L, "1 Number: Beta");
	return 3;
    }
// 	if(func == l_pend)
// 	{
// 		lua_pushstring(L, "Calculate the Nth derivative of the path energy.");
// 		lua_pushstring(L, "1 Integer: The derivative, 0 = energy, 1 = slope of energy, 2 = acceleration of energy, ...");
// 		lua_pushstring(L, "2 Tables: The Nth derivatives of the energy, the normalized location along the path between 0 and 1.");
// 		return 3;
// 	}

    if(func == _l_getCoordinateSystems)
    {
	lua_pushstring(L, "Get available coordinate systems.");
	lua_pushstring(L, "");
	lua_pushstring(L, "1 Table: Table of strings, coordinate system names. Table = {\"## return table.concat(MEP.new():coordinateSystems(), [[\", \"]]) ##\"}");
	return 3;
    }

    if(func == _l_setCoordinateSystem)
    {
	lua_pushstring(L, "Set the internal coordinate system for points and sites. This can be changed during the calculation.");
	lua_pushstring(L, "1 String followed by a nil, a Table of Integers or an Integer followed by a nil, a Table of integers or an Integer: The string defines the coordinate system, the second and third blocks specify point and site informaton. For each block, if a nil is supplied then all the existing points or sites are set to the new coordinate system. If a table is provided then the 2 integers set the start and end range for the operation. If a single integer is provided then that point or site is set. Example: <pre>mep:setCoordinateSystem(\"Cartesian\", nil, {2,4})\nmep:setCoordinateSystem(\"Spherical\")\nmep:setCoordinateSystem(\"Canonical\", 1, 2)</pre>The first example sets the coordinate system to \"Cartesian\" for all points and for all sites from 2 to 4. The second example sets the coordinate system to \"Spherical\" for all points and all sites. The third example sets the coordinate system to \"Canonical\" for the second site of the first point.");
	lua_pushstring(L, "");
	return 3;
    }

    if(func == _l_getCoordinateSystem)
    {
	lua_pushstring(L, "Get the internal coordinate system.");
	lua_pushstring(L, "A nil, a Table of Integers or an Integer followed by a nil, a Table of integers or an Integer: The 1 blocks of input specify the point and site elements or ranges that are being queried. For single elements a single value will be returned, for ranges, a table of names will be returned. Nil values are interpreted as full ranges. ");
	lua_pushstring(L, "A table of table of strings, a table of strings or a single string: Coordinate system name for requested element(s).");
	return 3;
    }

    if(func == l_getep)
    {
	lua_pushstring(L, "Get the scale factor used in numerical derivatives to calculate step size. By default this is 1e-3");
	lua_pushstring(L, "");
	lua_pushstring(L, "1 Number: Scale factor used in numerical derivatives.");
	return 3;
    }

    if(func == l_setep)
    {
	lua_pushstring(L, "Set the scale factor used in numerical derivatives to calculate step size. By default this is 1e-3");
	lua_pushstring(L, "1 Number: New scale factor used in numerical derivatives.");
	lua_pushstring(L, "");
	return 3;
    }

/* moved to lua
    if(func == l_us)
    {
	lua_pushstring(L, "Get a list of which points that are unique.");
	lua_pushstring(L, "Zero or 1 number, 0 or 1 table: tolerances used to determine equality. If a number is provided it will be used. The tolerance is radian difference that two \"equal\" vectors can differ by. If a table is provided then only the points in the table will be considered.");
	lua_pushstring(L, "1 Table: Indexes of unique points");
	return 3;
    }
*/

    if(func == l_getrfm)
    {
	lua_pushstring(L, "Get the max number of tries to advance in a direction before attempting a new direction in the relax algorithm.");
	lua_pushstring(L, "");
	lua_pushstring(L, "1 Integer: Max number of tries.");
	return 3;
    }
    if(func == l_setrfm)
    {
	lua_pushstring(L, "Set the max number of tries to advance in a direction before attempting a new direction in the relax algorithm.");
	lua_pushstring(L, "1 Integer: Max number of tries.");
	lua_pushstring(L, "");
	return 3;
    }

    if(func == l_asbs)
    {
	lua_pushstring(L, "Determine the angles between pairs of points for each site");
	lua_pushstring(L, "2 Integers: Point indexes");
	lua_pushstring(L, "1 Table: Angles between sites");
	return 3;
    }

    return LuaBaseObject::help(L);
}





static luaL_Reg m[128] = {_NULLPAIR128};
const luaL_Reg* MEP::luaMethods()
{
    if(m[127].name)return m;
    merge_luaL_Reg(m, LuaBaseObject::luaMethods());

    static const luaL_Reg _m[] =
	{
	    {"setInternalData", l_setdata},
	    {"getInternalData", l_getdata},
	    {"_addSite", l_addsite},
	    {"pathPointCount", l_ppc},
	    {"sites", l_getallsites},
	    {"siteCount", l_sitecount},
	    {"clearSites", l_clearsites},
	    {"clearPath", l_clearpath},
	    {"makeForcePerpendicularToSpins", l_projForcePerpSpins},
	    {"makeForcePerpendicularToPath", l_projForcePerpPath},
	    {"makeForceParallelToPath", l_projForcePath},
	    {"calculateEnergyGradients", l_calculateEnergyGradients},
	    {"resampleStateXYZPath", l_resampleStateXYZPath},
	    {"getPathEnergy", l_getpathenergy},
	    {"calculateEnergies", l_calculateEnergies},
	    {"calculatePointEnergy", l_calculateSingleEnergy},
	    {"gradient", l_getgradient},
	    {"spin", l_getsite},
	    // {"spinInCoordinateSystem", l_getsite},
	    {"setSpin", l_setsite},
	    // {"setSpinInCoordinateSystem", l_setsite},
	    {"applyForces", l_applyforces},
	    {"_nativeSpin", l_nativespin},
	    {"_addStateXYZ", l_addstatexyz},
	    {"_setImageSiteMobility", l_setimagesitemobility},
	    {"_getImageSiteMobility", l_getimagesitemobility},
	    {"_relaxSinglePoint_sd", l_relaxSinglePoint_sd_}, //internal method
	    {"_relaxSinglePoint_ex", l_relaxSinglePoint_ex_},
	    {"_hessianAtPoint", l_computepoint2deriv},
	    {"_gradAtPoint", l_computepoint1deriv},
	    {"_maximalPoints", l_maxpoints},
	    {"_randomize", l_randomize_},
	    {"internalCopyTo", l_internal_copyto},
	    {"absoluteDifference", l_absoluteDifference},
	    {"setBeta", l_setbeta},
	    {"beta", l_getbeta},
	    {"_pathEnergyNDeriv", l_pend_},
	    {"coordinateSystems", _l_getCoordinateSystems},
	    {"coordinateSystem", _l_getCoordinateSystem},
	    {"setCoordinateSystem", _l_setCoordinateSystem},
	    {"_convertCoordinateSystem", l_conv_cs},
	    {"epsilon", l_getep},
	    {"setEpsilon", l_setep},
	    //{"uniquePoints", l_us},
	    {"setRelaxDirectionFailMax", l_setrfm},
	    {"relaxDirectionFailMax", l_getrfm},
	    {"_slidePoint", l_slidePoint_},
	    {"_classifyPoint", l_cp_},
	    {"anglesBetweenPoints", l_asbs},
	    {"_computePointSecondDerivativeAB", l_computePointSecondDerivativeAB},
	    {"_resize", _l_resize},
	    {"_swap", _l_swap},
	    {"_copy", _l_copy},
	    {"_angleBetweenPointSite", _l_angle_between_pointsite},
	    {"_expensiveEnergyMinimization", _l_expensiveenergyminimization},
	    {"_expensiveEnergyMaximization", _l_expensiveenergymaximization},
	    {"_expensiveGradientMinimization", _l_expensivegradientminimization},
	    {"_classifyPoint", _l_classifypoint},
	    {"_findBestPath", _l_bestpath},
	    {"_interpolateBetweenCustomPoints", _l_int_cp},
            {"_rotatePathAboutBy", _l_rpab},
            {"_setSpinSystem", _l_setspinsystem},
            {"_getSpinSystem", _l_getspinsystem},
            {"_setEnergyFunction", _l_setEnergyFunction},
            {"_getEnergyFunction", _l_getEnergyFunction},
            {"findCriticalIslands", _l_findlowgradislands},
	    {NULL, NULL}
	};
    merge_luaL_Reg(m, _m);
    m[127].name = (char*)1;
    return m;
}






extern "C"
{
    static int l_getmetatable(lua_State* L)
    {
	if(!lua_isstring(L, 1))
	    return luaL_error(L, "First argument must be a metatable name");
	luaL_getmetatable(L, lua_tostring(L, 1));
	return 1;
    }

    _MEP_API int lib_register(lua_State* L)
    {
	luaT_register<MEP>(L);

	lua_pushcfunction(L, l_getmetatable);
	lua_setglobal(L, "maglua_getmetatable");


	const char* s = __mep_luafuncs();

        luaL_dofile_mep_luafuncs(L);

	lua_pushnil(L);
	lua_setglobal(L, "maglua_getmetatable");


	return 0;
    }

    _MEP_API int lib_version(lua_State* L)
    {
	return __revi;
    }

    _MEP_API const char* lib_name(lua_State* L)
    {
#if defined NDEBUG || defined __OPTIMIZE__
	return "MEP";
#else
	return "MEP-Debug";
#endif
    }

    _MEP_API int lib_main(lua_State* L)
    {
	return 0;
    }
}
