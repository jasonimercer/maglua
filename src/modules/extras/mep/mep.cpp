#include "mep.h"
#include "mep_luafuncs.h"
#include "info.h"
#include "luamigrate.h"
#include <math.h>

#include <algorithm>
#include <numeric>

#include "ddd.h"

static const double seven_directions[7][3] = {
	{ 0, 0, 0},
	{ 0, 1, 0},
	{ 0,-1, 0},
	{ 0, 0, 1},
	{ 0, 0,-1},
	{ 1, 0, 0}, // last 2 aren't used 
	{-1, 0, 0}  // for spherical/canonical
};

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





// deterministic random number generator - don't want to interfere with
// the internal state of the C generator. Too lazy to use stateful rng
static unsigned long next = 1;


/* RAND_MAX assumed to be 32767 */
int myrand(void) {
	next = next * 1103515245 + 12345;
	return((unsigned)(next/65536) % 32768);
}
double myrandf()
{
	return (double)(myrand()) / 32767.0;
}

static double dot(const double* a, const double* b, int n = 3)
{
	double s = 0;
	for(int i=0; i<n; i++)
		s += a[i]*b[i];
	return s;
}

static double _magnitude(MEP::CoordinateSystem cs, const double* v)
{
	if(cs == MEP::Cartesian)
	{
		return sqrt(dot(v,v));
	}
	return v[0];
}
static double _magnitude(const double* v, MEP::CoordinateSystem cs)
{
	return _magnitude(cs,v);
}


void _setmagnitude(MEP::CoordinateSystem cs, double* v, double mag)
{
	if(cs == MEP::Cartesian)
	{
		double m = _magnitude(cs, v);
		if(m != 0)
		{
			v[0] *= mag/m;
			v[1] *= mag/m;
			v[2] *= mag/m;
		}
		else
		{
			v[0] = mag;
		}
	}
	else
	{
		v[0] = mag;
	}
}
static void _setmagnitude(double* v, double mag, MEP::CoordinateSystem cs)
{
	_setmagnitude(cs,v,mag);
}


static void _rotate(double* vec, int a, int b, int direction)
{
	double t = vec[a];
	vec[a] = vec[b];
	vec[b] = t;

	if(direction < 0)
		vec[a] *= -1.0;
	else
		vec[b] *= -1.0;
}

static void _squashRadialComponent(MEP::CoordinateSystem cs, double* vecs, int nvecs=1)
{
	if(cs == MEP::Cartesian)
		return;
	for(int i=0; i<nvecs; i++)
	{
		vecs[i*3] = 0;
	}
}

static char csRotatedType(MEP::CoordinateSystem cs)
{
	switch(cs)
	{
	case MEP::Undefined:
	case MEP::Cartesian:
	case MEP::Spherical:
	case MEP::Canonical:
		return ' ';
	case MEP::SphericalX:
	case MEP::CanonicalX:
		return 'X';
	case MEP::SphericalY:
	case MEP::CanonicalY:
		return 'Y';
	}
	return ' ';
}

static MEP::CoordinateSystem csRotate(MEP::CoordinateSystem cs, char r)
{
	if(cs == MEP::Spherical)
	{
		switch(r)
		{
		case 'x':
		case 'X':
			return MEP::SphericalX;
		case 'y':
		case 'Y':
			return MEP::SphericalY;
		}
	}
	if(cs == MEP::Canonical)
	{
		switch(r)
		{
		case 'x':
		case 'X':
			return MEP::CanonicalX;
		case 'y':
		case 'Y':
			return MEP::CanonicalY;
		}
	}
	return MEP::Undefined;
}

static MEP::CoordinateSystem csUnrotated(MEP::CoordinateSystem cs)
{
	switch(cs)
	{
	case MEP::Undefined:
	case MEP::Cartesian:
	case MEP::Spherical:
	case MEP::Canonical:
		return cs;
 	case MEP::SphericalX:
	case MEP::SphericalY:
		return MEP::Spherical;
	case MEP::CanonicalX:
	case MEP::CanonicalY:
		return MEP::Canonical;
	}
	return cs;
}

static void rotate(double* vec, char plane, int direction)
{
	switch(plane)
	{
	case 'x':
	case 'X':
		_rotate(vec, 1,2,direction);
		break;
	case 'y':
	case 'Y':
		_rotate(vec, 0,2,direction);
		break;
	case 'z':
	case 'Z':
		_rotate(vec, 0,1,direction);
		break;
	}
}

#define _convertCoordinateSystem(ss, ds, src, dest) _convertCoordinateSystem_(ss,ds,src,dest,__FILE__, __LINE__)

static void _convertCoordinateSystem_(
	MEP::CoordinateSystem src_cs, MEP::CoordinateSystem dest_cs,
	const double* src, double* dest, const char* file, unsigned int line)
{
	if(src_cs == dest_cs)
	{
		memcpy(dest, src, 3 * sizeof(double));
		return;
	}

	if(src_cs == MEP::Undefined || dest_cs == MEP::Undefined)
	{
		dest[0] = 0;
		dest[1] = 0;
		dest[2] = 0;
		return;
	}

	double temp[3];
	if(src == dest)
	{
		// copy to temp
		_convertCoordinateSystem_(src_cs, src_cs, src, temp, file, line);
		src = temp; // to make sure we're not stomping around like morons, overwriting parts of src with dest
	}

	char src_r = csRotatedType(src_cs);
	if(src_r != ' ') //then it's a rotated type
	{
		_convertCoordinateSystem_(csUnrotated(src_cs), MEP::Cartesian, src, dest, file, line);
		rotate(dest, src_r, -1); //remove rotation type
		_convertCoordinateSystem_(MEP::Cartesian, dest_cs, dest, dest, file, line);
	}

	char dest_r = csRotatedType(dest_cs);
	if(dest_r != ' ') // then it's a rotated type
	{
		if(src_cs == MEP::Cartesian)
		{
			// do the rotation
			memcpy(dest, src, sizeof(double)*3);
			rotate(dest, dest_r, 1); // apply rotation type
			_convertCoordinateSystem_(src_cs, csUnrotated(dest_cs), dest, dest, file, line);
			return;
		}
		else
		{
			double c[3];
			_convertCoordinateSystem_(src_cs, MEP::Cartesian, src, c, file, line);
			_convertCoordinateSystem_(MEP::Cartesian, dest_cs, c, dest, file, line);
			return;
		}
	}
		


	if(src_cs == MEP::Cartesian)
	{
		const double x = src[CARTESIAN_X];
		const double y = src[CARTESIAN_Y];
		const double z = src[CARTESIAN_Z];

		const double r = sqrt(x*x + y*y + z*z);
		const double p = atan2(y, x);
		const double t = _acos(z / r);

		if(dest_cs == MEP::Spherical)
		{
			dest[SPHERICAL_R] = r;
			dest[SPHERICAL_PHI] = p;
			dest[SPHERICAL_THETA] = t;
		}
		if(dest_cs == MEP::Canonical)
		{
			dest[CANONICAL_R] = r;
			dest[CANONICAL_PHI] = p;
			dest[CANONICAL_P] = cos(t);
		}
	}
	if(src_cs == MEP::Spherical)
	{
		if(dest_cs == MEP::Canonical)
		{
			dest[CANONICAL_R] = src[SPHERICAL_R];
			dest[CANONICAL_PHI] = src[SPHERICAL_PHI];
			dest[CANONICAL_P] = cos(src[SPHERICAL_THETA]);
		}
		if(dest_cs == MEP::Cartesian)
		{
			dest[CARTESIAN_X] = src[0] * cos( src[SPHERICAL_PHI] ) * sin( src[SPHERICAL_THETA] );
			dest[CARTESIAN_Y] = src[0] * sin( src[SPHERICAL_PHI] ) * sin( src[SPHERICAL_THETA] );
			dest[CARTESIAN_Z] = src[0] * cos( src[SPHERICAL_THETA] );
		}
	}
	if(src_cs == MEP::Canonical)
	{
		double r = src[CANONICAL_R];
		double phi = src[CANONICAL_PHI];
		double p = src[CANONICAL_P];

		while(p < -1 || p > 1)
		{
			while(p < -1)
				p += 4;
			if(p > 1)
			{
				p = 2.0 - p;
				phi = phi + M_PI; // flip phi
			}
		}

		while(phi < 0)
			phi += 2.0 * M_PI;
		while(phi >= 2.0 * M_PI)
			phi -= 2.0 * M_PI;
			  


		if(dest_cs == MEP::Spherical)
		{
			dest[SPHERICAL_R] = r;
			dest[SPHERICAL_PHI] = phi;
			dest[SPHERICAL_THETA] = _acos(p);
		}
		if(dest_cs == MEP::Cartesian)
		{
			const double t = _acosFL(p, file, line );
			dest[CARTESIAN_X] = r * cos( phi ) * sin( t );
			dest[CARTESIAN_Y] = r * sin( phi ) * sin( t );
			dest[CARTESIAN_Z] = r * p;
		}
	}
}

static void _normalizeTo(double* dest, const double* src, MEP::CoordinateSystem cs, const double length=1.0, const int n=3)
{
	if(cs == MEP::Cartesian)
	{
		double len = dot(src, src, n);
		if(len == 0)
		{
			dest[0] = length;
			for(int i=1; i<n; i++)
				dest[i] = 0;
		}
		else
		{
			const double iln = length/sqrt(len);
			for(int i=0; i<n; i++)
				dest[i] = iln * src[i];
		}
		return;
	}
	dest[0] = length;
	dest[1] = src[1];
	dest[2] = src[2];
}

static void _scaleFactors(MEP::CoordinateSystem cs, const double* vec, double* sf)
{
	switch(cs)
	{
	case MEP::Cartesian:
		sf[CARTESIAN_X] = 1;
		sf[CARTESIAN_Y] = 1;
		sf[CARTESIAN_Z] = 1;
		break;
	case MEP::Spherical:
	case MEP::SphericalX:
	case MEP::SphericalY:
		if(vec[SPHERICAL_R] == 0)
		{
			sf[SPHERICAL_R] = 1;
			sf[SPHERICAL_PHI] = 1;
			sf[SPHERICAL_THETA] = 1;
		}
		else
		{
			if(sin(vec[SPHERICAL_THETA]) == 0)
			{
				sf[SPHERICAL_R] = 1;
				sf[SPHERICAL_PHI] = 1;
				sf[SPHERICAL_THETA] = 1.0 / vec[SPHERICAL_R];
			}
			else
			{
				sf[SPHERICAL_R] = 1;
				sf[SPHERICAL_PHI] = 1/(sin(vec[SPHERICAL_THETA]) * vec[SPHERICAL_R]);
				sf[SPHERICAL_THETA] = 1/vec[SPHERICAL_R];
			}
		}
		break;
	case MEP::Canonical:
	case MEP::CanonicalX:
	case MEP::CanonicalY:
		if(vec[CANONICAL_R] == 0)
		{
			sf[CANONICAL_R] = 1;
			sf[CANONICAL_PHI] = 1;
			sf[CANONICAL_P] = 1;
			return;
		}

		if(vec[CANONICAL_P] * vec[CANONICAL_P] == 1)
		{
			sf[CANONICAL_R] = 1;   // definitely not zero like P
			sf[CANONICAL_PHI] = 1; // if P^2 is 1 then the PHI term is 1/0 = inf. Let's call it 1
			sf[CANONICAL_P] = 1;   // if P^2 is 1 then the P term is 1/(1/0)) = 1/inf = 0. Trying 1
			return;
		}

		const double r2 = vec[CANONICAL_R] * vec[CANONICAL_R];
		const double p2 = vec[CANONICAL_P] * vec[CANONICAL_P];

		sf[CANONICAL_R]   = 1;
		sf[CANONICAL_PHI] = 1/sqrt(r2*(1-p2));
		sf[CANONICAL_P]   = 1/sqrt(r2/(1-p2));
		break;
	}
}

static void _stepSize(MEP::CoordinateSystem cs, const double* vec, const double epsilon, double* h3)
{
	const double m = _magnitude(cs, vec);
	switch(cs)
	{
	case MEP::Cartesian:
		h3[CARTESIAN_X] = m*epsilon;
		h3[CARTESIAN_Y] = m*epsilon;
		h3[CARTESIAN_Z] = m*epsilon;
		break;
	case MEP::Spherical:
	case MEP::SphericalX:
	case MEP::SphericalY:
		h3[SPHERICAL_R] = m*epsilon;
		h3[SPHERICAL_PHI] = 2*M_PI*epsilon;
		h3[SPHERICAL_THETA] = M_PI*epsilon;
		break;
	case MEP::Canonical:
	case MEP::CanonicalX:
	case MEP::CanonicalY:
		h3[CANONICAL_R] = m*epsilon;
		h3[CANONICAL_PHI] = 2*M_PI*epsilon;
		h3[CANONICAL_P] =    2*epsilon;
		break;
	}	
}


static void _randomizeDirection(double* v, double m, MEP::CoordinateSystem cs)
{
	double a[3];
	_convertCoordinateSystem(cs, MEP::Cartesian, v, a);

	const double m1 = sqrt(dot(a,a,3));

	double nv[3];
	_normalizeTo(nv, a, cs);
			
	nv[0] += (myrandf() * m);
	nv[1] += (myrandf() * m);
	nv[2] += (myrandf() * m);
	
	_normalizeTo(a, nv, cs, m1);

	_convertCoordinateSystem(MEP::Cartesian, cs, a, v);
}

static void _scale(double* dest, const double* src, double s, MEP::CoordinateSystem cs, int n=3)
{
	if(cs == MEP::Cartesian)
	{
		for(int i=0; i<n; i++)
			dest[i] = src[i] * s;
	}
	else
	{
		if(dest != src)
			memcpy(dest, src, sizeof(double)*n);
		dest[0] *= s;
	}
}

// project a in the b direction
static void _project(double* dest, const double* a, const double* b, MEP::CoordinateSystem cs, int n=3)
{
	if(cs == MEP::Cartesian)
	{
		// printf("p a: %e %e %e\n", a[0], a[1], a[2]);
		// printf("p b: %e %e %e\n", b[0], b[1], b[2]);
		double ab = dot(a,b,n);
		double bb = dot(b,b,n);
		
		if(bb == 0)
			_scale(dest, b,     0, cs, n);
		else
			_scale(dest, b, ab/bb, cs, n);

		// printf("p p: %e %e %e\n", dest[0], dest[1], dest[2]);
	}
	else
	{
		if(n == 3)
		{
			double _dest[3];
			double _a[3];
			double _b[3];
			_convertCoordinateSystem(cs, MEP::Cartesian, a, _a);
			_convertCoordinateSystem(cs, MEP::Cartesian, b, _b);

			_project(_dest, _a, _b, MEP::Cartesian, 3);

			_convertCoordinateSystem(MEP::Cartesian, cs, _dest, dest);
		}
		else
		{
			fprintf(stderr, "Don't know how to deal with Spherical or Canonical when dimension != 3\n");
		}
	}
}

// negterm = -1 is standard rejection
// negterm = -2 negates the vector in the given direction
static void _reject(double* dest, const double* a, const double* b, MEP::CoordinateSystem cs, int n=3, double neg_term=-1.0)
{
	if(cs == MEP::Cartesian)
	{
		// printf("reject\n");
		double* _dest = new double[n];
		// printf("a: %e %e %e\n", a[0], a[1], a[2]);
		// printf("b: %e %e %e\n", b[0], b[1], b[2]);
		_project(_dest, a, b, cs, n);
		// printf("p: %e %e %e\n", _dest[0], _dest[1], _dest[2]);
		for(int i=0; i<n; i++)
			dest[i] = a[i] + neg_term * _dest[i];
		// printf("r: %e %e %e\n", dest[0], dest[1], dest[2]);
		delete [] _dest;
	}
	else
	{
		double _dest[3];
		double _a[3];
		double _b[3];
		_convertCoordinateSystem(cs, MEP::Cartesian, a, _a);
		_convertCoordinateSystem(cs, MEP::Cartesian, b, _b);
		
		_reject(_dest, _a, _b, MEP::Cartesian, neg_term);
		
		_convertCoordinateSystem(MEP::Cartesian, cs, _dest, dest);
	}
}

static double _angleBetween(const double* a, const double* b, MEP::CoordinateSystem cs1, MEP::CoordinateSystem cs2)
{
	double c1[3];
	double c2[3];
	
	_convertCoordinateSystem(cs1, MEP::Cartesian, a, c1);
	_convertCoordinateSystem(cs2, MEP::Cartesian, b, c2);

	double d1 = dot(c1,c1);
	double d2 = dot(c2,c2);

	const double c1c2 = dot(c1,c2);

	double ct = c1c2 / (sqrt(d1) * sqrt(d2));
	// floating point shenanigans can trigger the following
	if(ct < -1)
		ct = -1;
	if(ct > 1)
		ct = 1;

	const double tt = _acos(ct);

//	_D(FL, "a", a, 3);
//	_D(FL, "b", b, 3);
//	_D(FL, "t", tt);

	return tt;
}

static double _angleBetween(const double* a, const double* b, MEP::CoordinateSystem cs)
{
	return _angleBetween(a,b,cs,cs);
}

static void cross(const double* a, const double* b, double* c)
{
	c[0] = a[1]*b[2] - a[2]*b[1];
	c[1] = a[2]*b[0] - a[0]*b[2];
	c[2] = a[0]*b[1] - a[1]*b[0];
}

// input vector: a
// requires unit length normal vector: n
// input theta: t
// output vector: b
static void _rotateAboutBy(MEP::CoordinateSystem cs, const double* _a, const double* _n, const double t, double* _b)
{
	double a[3];
	double n[3];
	double b[3];

	_convertCoordinateSystem(cs, MEP::Cartesian, _a, a);
	_convertCoordinateSystem(cs, MEP::Cartesian, _n, n);

	const double ux = n[0];
	const double uy = n[1];
	const double uz = n[2];

	const double sx = a[0];
	const double sy = a[1];
	const double sz = a[2];
	
// 	printf("(%s:%i)\n", __FILE__, __LINE__);
// 	printf("rotate [%g %g %g] about [%g %g %g] by %g\n", a[0], a[1], a[2], n[0], n[1], n[2], t);
	
	// rotating based on Taylor, Camillo; Kriegman (1994). "Minimization on the Lie Group SO(3) and Related Manifolds". Technical Report. No. 9405 (Yale University).
	//
	// 	Here is the rotation matrix:
	const double cost = cos(t);
	const double sint = sin(t);
	
	const double R[3][3] = {
		{ cost + ux*ux * (1.0-cost),   ux*uy*(1.0-cost) - uz*sint, ux*uz * (1.0-cost) + uy * sint},
		{ uy*ux*(1.0-cost) + uz*sint,   cost + uy*uy * (1.0-cost), uy*uz * (1.0-cost) - ux * sint},
		{ uz*ux*(1.0-cost) - uy*sint, uz*uy*(1.0-cost) + ux*sint, cost + uz*uz * (1.0-cost)} };
	
	// now to multiply R * {sx,sy,sz}
	b[0] = R[0][0]*sx + R[1][0]*sy + R[2][0]*sz;
	b[1] = R[0][1]*sx + R[1][1]*sy + R[2][1]*sz;
	b[2] = R[0][2]*sx + R[1][2]*sy + R[2][2]*sz;
	
	_convertCoordinateSystem(MEP::Cartesian, cs, b, _b);
}


MEP::MEP()
	: LuaBaseObject(hash32(MEP::slineage(0)))
{
	ref_data = LUA_REFNIL;
	energy_ok = false;
	good_distances = false;
	beta = 0.1;
	relax_direction_fail_max = 4;
	epsilon = 1e-3;
	currentSystem = Cartesian;
}

MEP::~MEP()
{
	deinit();
	
}


void MEP::init()
{
}

void MEP::deinit()
{
	if(ref_data != LUA_REFNIL)
		luaL_unref(L, LUA_REGISTRYINDEX, ref_data);
	ref_data = LUA_REFNIL;

	state_xyz_path.clear();
	image_site_mobility.clear();
	sites.clear();
}


int MEP::luaInit(lua_State* L, int base)
{
	state_xyz_path.clear();
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
	
	encodeInteger(state_xyz_path.size(), b);
	for(int i=0; i<state_xyz_path.size(); i++)
	{
		encodeDouble(state_xyz_path[i], b);
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
	
	int n = decodeInteger(b);
	for(int i=0; i<n; i++)
	{
		state_xyz_path.push_back(decodeDouble(b));
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

void MEP::internal_copy_to(MEP* dest)
{
	dest->state_xyz_path.clear();

	for(int i=0; i<state_xyz_path.size(); i++)
	{
		dest->state_xyz_path.push_back(state_xyz_path[i]);
	}
	
	dest->image_site_mobility.clear();
	for(int i=0; i<image_site_mobility.size(); i++)
	{
		dest->image_site_mobility.push_back(image_site_mobility[i]);
	}

	//dest->state_xyz_path = state_xyz_path;
	dest->currentSystem = currentSystem;
	dest->sites = sites;
	dest->path_tangent = path_tangent;
	dest->force_vector = force_vector;
	dest->energies = energies;
	dest->energy_ok = false;
	dest->beta = beta;
}

const char* MEP::nameOfCoordinateSystem(CoordinateSystem s)
{
	switch(s)
	{
	case Undefined: return "Undefined";
	case Cartesian: return "Cartesian";
	case Spherical: return "Spherical";
	case Canonical: return "Canonical";
	case SphericalX: return "SphericalX";
	case SphericalY: return "SphericalY";
	case CanonicalX: return "CanonicalX";
	case CanonicalY: return "CanonicalY";
	}
	return 0;
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

int MEP::l_setCoordinateSystem(lua_State* L, int idx)
{
	const char* new_cs = lua_tostring(L, idx);
	if(new_cs == 0)
	{
		return luaL_error(L, "empty coordinate system");
	}
	CoordinateSystem newSystem = Undefined;
	for(int i=0; i<=6; i++)
	{
		CoordinateSystem j = (CoordinateSystem)i;
		if(strncasecmp(new_cs, nameOfCoordinateSystem(j), strlen(nameOfCoordinateSystem(j))) == 0)
		{
			newSystem = j;
		}
	}

	if(newSystem != Undefined)
	{
		if(newSystem != currentSystem)
		{
			const int ni = numberOfImages();
			const int ns = numberOfSites();

			for(int i=0; i<ni; i++)
			{
				for(int s=0; s<ns; s++)
				{
					const int k = (i*ns+s);

					double* v = &(state_xyz_path[k*3]);

					_convertCoordinateSystem(currentSystem, newSystem, v, v);
				}
			}
		}
		currentSystem = newSystem;
		return 0;
	}

	return luaL_error(L, "Unknown Coordinate System");
}

int MEP::l_getCoordinateSystem(lua_State* L)
{
	lua_pushstring(L, nameOfCoordinateSystem(currentSystem));
	return 1;
}

void MEP::convertCoordinateSystem(
	CoordinateSystem src_cs, CoordinateSystem dest_cs,
	const double* src, double* dest) const
{
	_convertCoordinateSystem(src_cs, dest_cs, src, dest);
}

double MEP::angleBetween(const double* v1, const double* v2, CoordinateSystem cs)
{
	_angleBetween(v1,v2,cs);
}

double MEP::vectorLength(const double* v1, CoordinateSystem cs)
{
	if(cs == Canonical || cs == Spherical)
		return v1[0];

	if(cs == Cartesian)
		return sqrt(dot(v1,v1));

	return 0;
}

double MEP::arcLength(const double* v1, const double* v2, CoordinateSystem cs)
{
	const double a = angleBetween(v1,v2,cs);
	const double r = vectorLength(v1,cs);

	const double c = 2.0 * M_PI * r;
	return c * (a / (2.0 * M_PI));
}





double MEP::absoluteDifference(MEP* other, int point, double& max_diff, int& max_idx)
{
	if(state_xyz_path.size() != other->state_xyz_path.size())
	{
// 		printf("s1.size() = %i, s2.size() = %i\n", state_xyz_path.size(), other->state_xyz_path.size());
		return -1;
	}

	double d = 0;
	max_diff = 0;
	max_idx = 0;

	int min = 0;
	int max = state_xyz_path.size();

	if(point >= 0)
	{
		min =  point    * numberOfSites() * 3;
		max = (point+1) * numberOfSites() * 3;
	}

	for(int i=min; i<max; i+=3)
	{
		const double diff = _angleBetween(&(state_xyz_path[i]), &(other->state_xyz_path[i]), currentSystem, other->currentSystem);

		if(diff > max_diff)
		{
			max_diff = diff;
			max_idx = i/3;
		}
		
		d += diff;
	}

	return d;
}



void MEP::randomize(const double magnitude)
{
	double* v = &state_xyz_path[0];

	const int num_sites = numberOfSites();
	double nv[3];

	const int ni = numberOfImages(); //images
	const int ns = numberOfSites();
	
	for(int i=0; i<ni; i++)
	{
		for(int s=0; s<ns; s++)
		{
			const int k = (i*ns+s);
			
			_randomizeDirection( &(v[k*3]), magnitude * image_site_mobility[k], currentSystem);
		}
	}
}



double MEP::distanceBetweenPoints(int p1, int p2, int site)
{
	const int num_sites = sites.size() / 3;

	const double* v1 = &state_xyz_path[p1 * num_sites * 3 + site * 3];
	const double* v2 = &state_xyz_path[p2 * num_sites * 3 + site * 3];
	
	const double d = angleBetween(v1, v2, currentSystem);

// 	printf("v1: %g %g %g\n", v1[0], v1[1], v1[2]);
// 	printf("v2: %g %g %g\n", v2[0], v2[1], v2[2]);
// 	printf("abp %i %i %g\n", p1, p2,  d);
	
	return d;
}

double MEP::distanceBetweenHyperPoints(int p1, int p2)
{
	double d = 0;
	int n = sites.size() / 3;
	for(int i=0; i<n; i++)
	{
		const double dbp = distanceBetweenPoints(p1, p2, i);
		d += dbp;
	}
	return d;
}

void MEP::interpolatePoints(const int p1, const int p2, const int site, const double _ratio, vector<double>& dest, const double rjitter)
{
	const int num_sites = sites.size() / 3;
	const double* v1 = &state_xyz_path[p1 * num_sites * 3 + site * 3];
	const double* v2 = &state_xyz_path[p2 * num_sites * 3 + site * 3];

	//printf("p1 %i p2 %i site %i ratio %g\n", p1,p2,site,_ratio);
	
	double nv1[3];
	double nv2[3];

	double _nv1[3];
	double _nv2[3];

	double mobility = 1.0;
	
	const double m1 = getImageSiteMobility(p1, site);
	const double m2 = getImageSiteMobility(p2, site);
	
	if(m1 < mobility)
		mobility = m1;
	if(m2 < mobility)
		mobility = m2;
	
	_normalizeTo(_nv1, v1, currentSystem);
	_normalizeTo(_nv2, v2, currentSystem);
	
	_convertCoordinateSystem(currentSystem, Cartesian, _nv1, nv1);
	_convertCoordinateSystem(currentSystem, Cartesian, _nv2, nv2);

// 	double rjitter = (myrandf() * jitter*2 - jitter) * mobility;

	double ratio = _ratio * (1.0 + rjitter * mobility);
	
	double a = angleBetween(nv1, nv2, Cartesian);

//	printf("%g %g %g     %g %g %g\n", v1[0], v1[1], v1[2],    v2[0], v2[1], v2[2]);
	
	if(a == 0 || ratio == 0) //then  no need to interpolate
	{
		dest.push_back(v1[0]);
		dest.push_back(v1[1]);
		dest.push_back(v1[2]);		
		return;
	}
	
	double norm[3];

// 	printf("(%s:%i)  nv1(%i): [%g, %g, %g]\n", __FILE__, __LINE__, p1, nv1[0], nv1[1], nv1[2]);
// 	printf("(%s:%i)  nv2(%i): [%g, %g, %g]\n", __FILE__, __LINE__, p2, nv2[0], nv2[1], nv2[2]);
	
// 	printf(">>> %f\n", fabs(a - 3.1415926538979));
	
	if(fabs(a - 3.1415926538979) < 1e-8) //then colinear, need a random ortho vector
	{
		double t[3]; 
		t[0] = -nv2[2]; // notice the coord twiddle
		t[1] = -nv2[0]; 
		t[2] =  nv2[1];
		cross(nv1,t,norm);

// 		printf("(%s:%i)\n",  __FILE__, __LINE__);
		if(dot(norm, norm) == 0) //then
		{
// 			printf("(%s:%i) RAND!!! \n",  __FILE__, __LINE__);
			t[0] = myrandf()*2.0-1.0;
			t[1] = myrandf()*2.0-1.0;
			t[2] = myrandf()*2.0-1.0;
			cross(nv1,t,norm); //assuming this works
		}
	}
	else
	{
// 		printf("(%s:%i)\n",  __FILE__, __LINE__);
		cross(nv1,nv2,norm);
	}

	_normalizeTo(norm, norm, Cartesian);

// 	normalizeTo(norm, norm);
// 	printf("(%s:%i) norm [%e %e %e]\n", __FILE__, __LINE__, norm[0], norm[1], norm[2]);
	
	
	double res[3];

	
	_rotateAboutBy(Cartesian, nv1, norm, -a*ratio, res);
	
	_normalizeTo(res, res, Cartesian, _magnitude(v1, currentSystem));
	
	_convertCoordinateSystem(Cartesian, currentSystem, res, nv1);

	dest.push_back(nv1[0]);
	dest.push_back(nv1[1]);
	dest.push_back(nv1[2]);
}



void MEP::interpolateHyperPoints(const int p1, const int p2, const double ratio, vector<double>& dest, const double jitter)
{
	int n = numberOfSites();
	//printf("(%s:%i) %i\n", __FILE__, __LINE__, n);
	for(int i=0; i<n; i++)
	{
		double rjitter = (myrandf() * jitter*2 - jitter);
		interpolatePoints(p1,p2,i,ratio,dest, rjitter);
	}
}


// get interval that bounds the distance
static int get_interval(const vector<double>& v, double x, double& ratio)
{
// 	printf("x = %f\n", x);	
	for(unsigned int i=0; i<v.size(); i++)
	{
		const double y = v[i];
// 		printf(">> %i %g %g\n", i,x,y);
		if(x < y)
		{
// 			printf("y: %g\n", y);
			ratio = x/y;
// 			printf("ratio: %g\n", ratio);
			return i;
		}
		x -= y;
	}
	
	// clip to end
	ratio = 1;
	return v.size()-1;
	
// 	return -1;
}

void MEP::printState()
{
	printf("State Path\n");
	
	for(int i=0; i<state_xyz_path.size(); i+=3)
	{
		const int k = i/3;
		printf("%3i\t%g\t%g\t%g   m=%g\n", k, state_xyz_path[i+0], state_xyz_path[i+1],state_xyz_path[i+2], image_site_mobility[k]);
	}
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

#include <algorithm>    // std::sort
int MEP::resampleStateXYZPath(lua_State* L)
{
	vector<double> points;
	
	double jitter = 0; //percent
	if(lua_isnumber(L, 3))
		jitter = lua_tonumber(L, 3);
	
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
			return luaL_error(L, "resampleStateXYZPath requires a number or a table of numbers");
		}
	}
	
	
	std::sort(points.begin(), points.end());

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
	vector<double> new_state_xyz_path;

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
			interpolateHyperPoints(j, j+1, ratio, new_state_xyz_path, jitter);
		}
		else
		{
			int nos = numberOfSites();
			for(int k=0; k<nos*3; k++)
			{
				new_state_xyz_path.push_back(0);
			}
			/*
			printf("failed to get interval\n");
			printf("  total distance: %g\n", distance);
			printf("  requested position: %g\n", points[i]);
			*/
		}
	}
	
	state_xyz_path.clear();
	for(unsigned int i=0; i<new_state_xyz_path.size(); i++)
	{
		state_xyz_path.push_back(new_state_xyz_path[i]);
	}
	
	// now we need to update mobility factors. The only case we will cover
	// is non-unitary endpoints.
	vector<double> first_mobility;// = image_site_mobility[0];
	vector<double> last_mobility;// = image_site_mobility[image_site_mobility.size()-1];
	
	const int ns = numberOfSites();
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
	return 0;
}

#if 0
int MEP::resampleStateXYZPath(lua_State* L)
{
//  	printf("resample from %i\n", state_xyz_path.size());

	int new_num_images = lua_tointeger(L, 2);
	
	double 
	
	double jitter = 0; //percent
	if(lua_isnumber(L, 3))
		jitter = lua_tonumber(L, 3);

	if(new_num_images < 2)
	{
		return luaL_error(L, "attempting to resample path to less than 2 images");
	}
	
// 	printf("pre-resample\n");
// 	printState();
	
	vector<double> new_state_xyz_path;
	const int num_points = numberOfSites();
	double distance = calculateD12(); //total distance


	
	// endpoints are fixed for resample path start
// 	for(int i=0; i<sites.size(); i++)
// 	{
// 		new_state_xyz_path.push_back( state_xyz_path[i] );
// 	}
// 	printf("distance = %f\n", distance);
	const double interval = distance / ((double)(new_num_images-1));
	double ratio;
	
	for(int i=0; i<new_num_images; i++)
	{
		int j = get_interval(d12, interval * i, ratio);
		if(j != -1)
		{
// 			printf("range[ %i:%i]  ratio: %e\n", j, j+1, ratio); 
			interpolateHyperPoints(j, j+1, ratio, new_state_xyz_path, jitter);
		}
		else
		{
			printf("failed to get interval\n");
			printf("  total distance: %g\n", distance);
			printf("  requested position: %g\n", interval * i);
		}
	}
	
	// endpoints are fixed for resample path end
// 	const int last_set_idx = state_xyz_path.size() - sites.size();
// 	for(int i=0; i<sites.size(); i++)
// 	{
// 		//printf("last: %f\n", state_xyz_path[last_set_idx + i]);
// 		new_state_xyz_path.push_back( state_xyz_path[last_set_idx + i] );
// 	}
	
	
	state_xyz_path.clear();
	for(unsigned int i=0; i<new_state_xyz_path.size(); i++)
	{
		state_xyz_path.push_back(new_state_xyz_path[i]);
	}
	
	// now we need to update mobility factors. The only case we will cover
	// is non-unitary endpoints.
	vector<double> first_mobility;// = image_site_mobility[0];
	vector<double> last_mobility;// = image_site_mobility[image_site_mobility.size()-1];
	
	const int ns = numberOfSites();
// 	new_num_images
	
	for(int i=0; i<ns; i++)
	{
		first_mobility.push_back( image_site_mobility[i] );
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
		image_site_mobility.push_back(last_mobility[i]);
	}
	
	energy_ok = false; //need to recalc energy
	good_distances = false;
	
// 	printf("post-resample\n");
// 	printState();

// 	printf("resample to %i new_images -- (%i)\n", new_num_images, state_xyz_path.size() / (numberOfSites() * 3));
	
	return 0;
}
#endif


void MEP::addSite(int x, int y, int z)
{
	sites.push_back(x);
	sites.push_back(y);
	sites.push_back(z);
	energy_ok = false; //need to recalc energy
}

//project gradients onto vector perpendicular to spin direction
void MEP::projForcePerpSpins(lua_State* L, int get_index, int set_index, int energy_index)
{
	if(currentSystem == Cartesian) // only tricky one right now
	{
		double proj[3];
		for(int i=0; i<state_xyz_path.size() / 3; i++)
		{
			double* force = &force_vector[i*3];
			double* v = &state_xyz_path[i*3];
			
			const double bb = dot(v, v);
			if(bb == 0)
			{
				proj[0] = 0;
				proj[1] = 0;
				proj[2] = 0;
			}
			else
			{
				const double ab = dot(force, v);
				proj[0] = v[0] * ab/bb;
				proj[1] = v[1] * ab/bb;
				proj[2] = v[2] * ab/bb;
			}
			
			force[0] -= proj[0];
			force[1] -= proj[1];
			force[2] -= proj[2];
		}
	}
	else
	{
		for(int i=0; i<state_xyz_path.size() / 3; i++)
		{
			force_vector[i*3] = 0; // remove radial force
		}
	}
}

int MEP::applyForces(lua_State* L)
{
	if(force_vector.size() != state_xyz_path.size())
		return luaL_error(L, "forces are not computed or size mismatch");
	
	const int num_sites = numberOfSites();
	// double absMovement = 0;
	// double maxMovement = 0;
	
	const int ni = numberOfImages();
	const int ns = numberOfSites();
	
	// end points are no longer hardcoded fixed. 
	for(int i=0; i<ni; i++)
	{
		for(int s=0; s<ns; s++)
		{
			const int k = (i*ns+s);

			double orig[3];
			double* v = &(state_xyz_path[k*3]);

			memcpy(orig, v, sizeof(double)*3);

			const double m1 = _magnitude(v, currentSystem);

			const double dC1 = force_vector[k*3+0] * image_site_mobility[k];
			const double dC2 = force_vector[k*3+1] * image_site_mobility[k];
			const double dC3 = force_vector[k*3+2] * image_site_mobility[k];
			
			v[0] -= dC1;
			v[1] -= dC2;
			v[2] -= dC3;

			// printf("v = %g %g %g, dd = %g %g %g\n", v[0], v[1], v[2],dC1, dC2, dC3);
			_normalizeTo(v, v, currentSystem, m1);
		}
	}
	
	return 0;
}
	


// this isn't used: 
void MEP::calculateOffsetVector(double* vec, const int p1, const int p2)
{
	const int ss = sites.size();
	for(int i=0; i<ss; i++)
	{
		vec[i] = state_xyz_path[p1*ss+i] - state_xyz_path[p2*ss+i];
	}
}

void MEP::computeTangent(const int p1, const int p2, const int dest)
{
	double a[3];
	double b[3];
	double c[3];
	const int s = sites.size();
	
	// compute difference
 	for(int i=0; i<s; i+=3)
	{
		_convertCoordinateSystem(currentSystem, Cartesian, &state_xyz_path[i + p1*s], a);
		_convertCoordinateSystem(currentSystem, Cartesian, &state_xyz_path[i + p2*s], b);

		for(int j=0; j<3; j++)
			c[j] = a[j] - b[j];

		_normalizeTo(c, c, Cartesian, 1);
		_convertCoordinateSystem(Cartesian, currentSystem, c, & path_tangent[i + dest*s]);
	}

}


void MEP::projForcePerpPath(lua_State* L, int get_index, int set_index, int energy_index) //project gradients onto vector perpendicular to path direction
{
	path_tangent.clear();
	path_tangent.resize(state_xyz_path.size());

	const int s = numberOfSites();

	computeTangent(0, 1, 0);
	for(int i=1; i<s-1; i++)
	{
		computeTangent(i-1,i+1,i);
	}
	computeTangent(s-2, s-1, s-1);

	const int ss = sites.size(); // point size
	const int nn = numberOfImages(); // number of points

	if(currentSystem == Cartesian)
	{
		// Cartesian values
		double* proj_c = new double[ss];
		double* force_c = new double[ss];
		double* v_c = new double[ss];
		
		for(int i=0; i<nn; i++)
		{
			double* force = &force_vector[i*ss];
			double* v = &path_tangent[i*ss];
			
			for(int j=0; j<ss; j+=3)
			{
				_convertCoordinateSystem(currentSystem, Cartesian, &force[j], &force_c[j]);
				_convertCoordinateSystem(currentSystem, Cartesian, &v[j], &v_c[j]);
			}
			
			const double bb = dot(v_c, v_c, ss);
			if(bb == 0)
			{
				for(int i=0; i<ss; i++)
				proj_c[i] = 0;
			}
			else
			{
				const double ab = dot(force_c, v_c, ss);
				for(int i=0; i<ss; i++)
					proj_c[i] = v_c[i] * ab/bb;;
			}
			
			for(int i=0; i<ss; i++)
				force_c[i] -= proj_c[i];
			
			for(int j=0; j<ss; j+=3)
			{
				_convertCoordinateSystem(Cartesian, currentSystem, &force_c[j], &force[j]);
			}
		}
		
		delete [] force_c;
		delete [] v_c;
		delete [] proj_c;
	}
}



void MEP::projForcePath(lua_State* L, int get_index, int set_index, int energy_index) //project gradients onto vector perpendicular to path direction
{
	path_tangent.clear();
	path_tangent.resize(state_xyz_path.size());

	const int s = numberOfSites();

	computeTangent(0, 1, 0);
	for(int i=1; i<s-1; i++)
	{
		computeTangent(i-1,i+1,i);
	}
	computeTangent(s-2, s-1, s-1);

	const int ss = sites.size(); // point size
	const int nn = numberOfImages(); // number of points

	// Cartesian values
	double* proj_c = new double[ss];
	double* force_c = new double[ss];
	double* v_c = new double[ss];

	for(int i=0; i<nn; i++)
	{
		double* force = &force_vector[i*ss];
		double* v = &path_tangent[i*ss];
		
		for(int j=0; j<ss; j+=3)
		{
			_convertCoordinateSystem(currentSystem, Cartesian, &force[j], &force_c[j]);
			_convertCoordinateSystem(currentSystem, Cartesian, &v[j], &v_c[j]);
		}

		const double bb = dot(v_c, v_c, ss);
		if(bb == 0)
		{
			for(int i=0; i<ss; i++)
				proj_c[i] = 0;
		}
		else
		{
			const double ab = dot(force_c, v_c, ss);
			for(int i=0; i<ss; i++)
				proj_c[i] = v_c[i] * ab/bb;;
		}
		
		for(int i=0; i<ss; i++)
			force_c[i] = proj_c[i];

		for(int j=0; j<ss; j+=3)
		{
			_convertCoordinateSystem(Cartesian, currentSystem, &force_c[j], &force[j]);
		}
	}

	delete [] force_c;
	delete [] v_c;
	delete [] proj_c;


}

// back up old sites so we can restore after
void MEP::saveConfiguration(lua_State* L, int get_index, vector<double>& buffer)
{
	buffer.clear();
	const int num_sites = numberOfSites();
	
	for(int s=0; s<num_sites; s++)
	{
		getSiteSpin(L, get_index, &sites[s*3+0], buffer);
	}
}

// restore state
void MEP::loadConfiguration(lua_State* L, int set_index, vector<double>& buffer)
{
	const int num_sites = numberOfSites();

	for(int s=0; s<num_sites; s++)
	{
		setSiteSpin(L, set_index, &sites[s*3], &buffer[s*3]);
	}
	


}

	
void MEP::getSiteSpin(lua_State* L, int get_index, int* site3, double* m3)
{
	double t[3];

	lua_pushvalue(L, get_index);
	lua_pushinteger(L, sites[0]+1);
	lua_pushinteger(L, sites[1]+1);
	lua_pushinteger(L, sites[2]+1);
	lua_call(L, 3, 3);

	t[0] = lua_tonumber(L, -3);
	t[1] = lua_tonumber(L, -2);
	t[2] = lua_tonumber(L, -1);
	
	_convertCoordinateSystem(Cartesian, currentSystem, t, m3);

	lua_pop(L, 3);	
}

void MEP::getSiteSpin(lua_State* L, int get_index, int* site3, vector<double>& v)
{
	double m3[3];
	getSiteSpin(L, get_index, site3, m3);
	v.push_back(m3[0]);
	v.push_back(m3[1]);
	v.push_back(m3[2]);
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

void MEP::setSiteSpin(lua_State* L, int set_index, int* site3, double* m3)
{
	double t[3];
	_convertCoordinateSystem(currentSystem, Cartesian, m3, t);

	lua_pushvalue(L, set_index);
	lua_pushinteger(L, site3[0]+1);
	lua_pushinteger(L, site3[1]+1);
	lua_pushinteger(L, site3[2]+1);
	lua_pushnumber(L, t[0]);
	lua_pushnumber(L, t[1]);
	lua_pushnumber(L, t[2]);
	lua_call(L, 6, 1);
	
	if(lua_toboolean(L, -1)) // then a change happened
	{
		energy_ok = false; //need to recalc energy
		good_distances = false;
	}
}

void MEP::setAllSpins(lua_State* L, int set_index, double* m)
{
	int num_sites = numberOfSites();
	for(int s=0; s<num_sites; s++)
		setSiteSpin(L, set_index, &sites[s*3], &m[s*3]);
}

void MEP::getAllSpins(lua_State* L, int get_index, double* m)
{
	int num_sites = numberOfSites();
	for(int s=0; s<num_sites; s++)
		getSiteSpin(L, get_index, &sites[s*3], &m[s*3]);
}

void MEP::setSiteSpin(lua_State* L, int set_index, int* site3, double v1, double v2, double v3)
{
	double m3[3];
	m3[0] = v1;
	m3[1] = v2;
	m3[2] = v3;
	
	setSiteSpin(L, set_index, site3, m3);
}


double MEP::getEnergy(lua_State* L, int energy_index)
{
	lua_pushvalue(L, energy_index);
	lua_call(L, 0, 1);
	double e = lua_tonumber(L, -1);
	lua_pop(L, 1);
	return e;
}


int MEP::calculateEnergies(lua_State* L, int get_index, int set_index, int energy_index)
{
	if(energy_ok)
		return 0;
		
	energies.clear();
	const int num_sites = numberOfSites();
	const int path_length = numberOfImages();

	// back up old sites so we can restore after
	vector<double> cfg;
	saveConfiguration(L, get_index, cfg);


	for(int p=0; p<path_length; p++)
	{
		double e = 0;
		for(int s=0; s<num_sites; s++)
		{
			setSiteSpin(L, set_index, &sites[s*3], &state_xyz_path[p * sites.size() + s*3]);
		}

		energies.push_back(getEnergy(L, energy_index));
	}

	loadConfiguration(L, set_index, cfg);
	
	energy_ok = true;
	return 0;
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


int MEP::calculatePathEnergyNDeriv(lua_State* L, int get_index, int set_index, int energy_index, int n, vector<double>& nderiv)
{
	calculateEnergies(L, get_index, set_index, energy_index); //always returns 0

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
	
	calculatePathEnergyNDeriv(L, get_index, set_index, energy_index, n-1, nmoDeriv);

	listDerivN(nderiv, nmoDeriv);
	
	return 0;
}


static void arrayCopyWithElementChange(MEP::CoordinateSystem cs, double* dest, double* src, int element, double delta, int n)
{
	memcpy(dest, src, sizeof(double)*n);
	dest[element] += delta;
}

static void arrayCopyWithElementsChange(MEP::CoordinateSystem cs, double* dest, double* src, int* directions, const double _dd, double* scale, int n)
{
	for(int i=0; i<n; i++)
	{
		const double dd = _dd * scale[i];
		const double* d = seven_directions[ directions[ i ] ];
		
		dest[i*3 + 0] = src[i*3 + 0] + d[0] * dd;
		dest[i*3 + 1] = src[i*3 + 1] + d[1] * dd;
		dest[i*3 + 2] = src[i*3 + 2] + d[2] * dd;
	}

}


static void arrayCopyWithElementsChange(MEP::CoordinateSystem cs, double* dest, double* src, int* directions, double* scale, int n)
{
	for(int i=0; i<n; i++)
	{
		const double* d = seven_directions[ directions[ i ] ];
		
		dest[i*3 + 0] = src[i*3 + 0] + d[0] * scale[i*3 + 0];
		dest[i*3 + 1] = src[i*3 + 1] + d[1] * scale[i*3 + 1];
		dest[i*3 + 2] = src[i*3 + 2] + d[2] * scale[i*3 + 2];
	}

}


static void rescale_vectors(MEP::CoordinateSystem cs, double* vecs, double* mags, int n)
{
	for(int i=0; i<n; i++)
	{
		_setmagnitude(cs, &vecs[i*3], mags[i]);
	}
}

double MEP::computePointSecondDerivativeAB(lua_State* L, int p, int set_index, int get_index, int energy_index, int c1, int c2, double _dc1, double _dc2)
{
	// back up old sites so we can restore after
	vector<double> cfg;
	saveConfiguration(L, get_index, cfg);
	const int num_sites = numberOfSites();
	double* vec = &state_xyz_path[p*num_sites*3];
	double result;
	double* state = new double[num_sites * 3];
	double e1,e2,e3,e4;
	double d1,d2;

	const int c1m3 = c1 % 3;
	const int c2m3 = c2 % 3;

	const int site1 = c1 - c1m3;
	const int site2 = c2 - c2m3;

	double stepSize1[3];
	double stepSize2[3];

	_stepSize(currentSystem, &(vec[site1]), epsilon, stepSize1);
	_stepSize(currentSystem, &(vec[site2]), epsilon, stepSize2);

	double dx1 = stepSize1[c1m3];
	double dx2 = stepSize2[c2m3];
	
	if(_dc1 > 0)
		dx1 = _dc1;
	if(_dc2 > 0)
		dx2 = _dc2;

	// calc upper deriv energies
	arrayCopyWithElementChange(currentSystem, state,   vec, c1, dx1, num_sites * 3);
	arrayCopyWithElementChange(currentSystem, state, state, c2, dx2, num_sites * 3);
	setAllSpins(L, set_index, state);
	e1 = getEnergy(L, energy_index);
	
	arrayCopyWithElementChange(currentSystem, state,   vec, c1, dx1, num_sites * 3);
	arrayCopyWithElementChange(currentSystem, state, state, c2,-dx2, num_sites * 3);
	setAllSpins(L, set_index, state);
	e2 = getEnergy(L, energy_index);
		
	// calc lower deriv energies
	arrayCopyWithElementChange(currentSystem, state,   vec, c1,-dx1, num_sites * 3);
	arrayCopyWithElementChange(currentSystem, state, state, c2, dx2, num_sites * 3);
	setAllSpins(L, set_index, state);
	e3 = getEnergy(L, energy_index);
	
	arrayCopyWithElementChange(currentSystem, state,   vec, c1,-dx1, num_sites * 3);
	arrayCopyWithElementChange(currentSystem, state, state, c2,-dx2, num_sites * 3);
	setAllSpins(L, set_index, state);
	e4 = getEnergy(L, energy_index);
	
	double diff_e1_e2 = (e1 - e2);
	double diff_e3_e4 = (e3 - e4);

	const double dd1 = diff_e1_e2 / (2.0 * dx2);
	const double dd2 = diff_e3_e4 / (2.0 * dx2);

	result = (dd1 - dd2) / (2.0 * dx1);

	loadConfiguration(L, set_index, cfg);	

	delete [] state;	

	return result;
}


// partial derivs
void MEP::computePointSecondDerivative(lua_State* L, int p, int set_index, int get_index, int energy_index, double* derivsAB)
{
	const int num_sites = numberOfSites();
	int deriv_pos = 0;
	for(int c1=0; c1<num_sites*3; c1++)
	{
		for(int c2=0; c2<num_sites*3; c2++)
		{
			derivsAB[deriv_pos] = computePointSecondDerivativeAB(L, p, set_index, get_index, energy_index, c1, c2);
			deriv_pos++;
		}
	}
}

void MEP::computePointFirstDerivative(lua_State* L, int p, int set_index, int get_index, int energy_index, double* d)
{
	const int num_sites = numberOfSites();
	for(int c=0; c<num_sites*3; c++)
	{
		d[c] = computePointFirstDerivativeC(L, p, set_index, get_index, energy_index, c);
		
	}
}

void MEP::computeVecFirstDerivative(lua_State* L, double* vec, int set_index, int get_index, int energy_index, double* d)
{
	const int num_sites = numberOfSites();
	for(int c=0; c<num_sites*3; c++)
	{
		d[c] = computeVecFirstDerivativeC(L, vec, set_index, get_index, energy_index, c);
	}
}

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


double MEP::computeVecFirstDerivativeC(lua_State* L, double* vec, int set_index, int get_index, int energy_index, int c1)
{
	// back up old sites so we can restore after
	vector<double> cfg;
	saveConfiguration(L, get_index, cfg);
	const int num_sites = numberOfSites();
	double result;
	double* state = new double[num_sites * 3];
	double* mags = new double[num_sites];
	double e1,e2;
	double d1;

	const int c1m3 = c1 % 3;

	const int site1 = c1 - c1m3;

	double scaleFactors1[3];

	double stepSize1[3];

	//_scaleFactors(currentSystem, &(vec[site1]), scaleFactors1);

	_stepSize(currentSystem, &(vec[site1]), epsilon, stepSize1);

	
	//const double sf = scaleFactors1[c1m3];
	const double dx1 = stepSize1[c1m3]; // / sf;

	for(int i=0; i<num_sites; i++)
	{
		mags[i] = _magnitude(currentSystem, &(vec[i*3]));
	}

	//printf("D = %g, C = %d\n", dx1, c1);
	arrayCopyWithElementChange(currentSystem, state,   vec, c1, dx1, num_sites * 3);
	rescale_vectors(currentSystem, state, mags, num_sites);
	setAllSpins(L, set_index, state);
	e1 = getEnergy(L, energy_index);
	
	//print_vec(state, num_sites*3);
	//printf("e1: %g\n", e1);

	arrayCopyWithElementChange(currentSystem, state,   vec, c1,-dx1, num_sites * 3);
	rescale_vectors(currentSystem, state, mags, num_sites);
	setAllSpins(L, set_index, state);
	e2 = getEnergy(L, energy_index);
		
	//print_vec(state, num_sites*3);
	//printf("e2: %g\n", e2);

   	d1 = (e1-e2);
	
	result  = d1;
	result /= (2.0 * dx1);
	
	loadConfiguration(L, set_index, cfg);	

	delete [] state;	
	delete [] mags;
	
	//printf("e1, e2 = %g, %g\n", e1, e2);

	return result;
}



double MEP::computePointFirstDerivativeC(lua_State* L, int p, int set_index, int get_index, int energy_index, int c1)
{
	const int num_sites = numberOfSites();
	double* vec = &state_xyz_path[p*num_sites*3];
	return computeVecFirstDerivativeC(L, vec, set_index, get_index, energy_index, c1);
}



#if 0
// converts an index to a value in an arbitrary base
// returns 1 on overflow or undeflow
static int _idx_2_number(int idx, int base, int* num, int max_digets)
{
	if(idx < 0)
		return 1;

	for(int i=0; i<max_digets; i++)
	{
		int n = idx % base;
		num[i] = n;
		idx = (idx - n) / base;
	}
	
	return idx;
}
	

static double vsum(vector<double>& v)
{
	double s = 0;
	for(unsigned int i=0; i<v.size(); i++)
		s += v[i];
	return s;
}

static void grad_check(double goal2, double min_grad2, int& num_steps, int& end_reason)
{
	if((goal2 > 0 && min_grad2 < goal2) || min_grad2 == 0) // then goal reached
	{
		num_steps = 0;
		end_reason = 1;
	}
}

// relax individual point. This is used to refine a maximal point
// expected on the stack:
// at 1, mep
// at 2, point number
// at 3, function get_site_ss1(x,y,z) return sx,sy,sz 
// at 4, function set_site_ss1(x,y,z, sx,sy,sz)  return something_changed
// at 5, function get_energy_ss1()
// at 6, step size
// at 7, num steps
// at 8, goal
// returns new step size coord 1, 2, 3 
int MEP::relaxSinglePoint(lua_State* L)
{
	const double step_up = 1.1;
	const double step_down = 0.5;

	const int num_sites = numberOfSites();
	const int path_length = numberOfImages();

	const char* end_reason[4] = {"finished number of steps", "reached goal", "", ""};
	int end_reason_i = 0;

	const int p = lua_tointeger(L, 2) - 1;
	if(p < 0 || p >= path_length)
		return luaL_error(L, "Point index out of bounds");

	const int get_index = 3;
	const int set_index = 4;
	const int energy_index = 5;

	if(!lua_isfunction(L, 3) || !lua_isfunction(L, 4) || !lua_isfunction(L, 5))
		return luaL_error(L, "3 functions expected - perhaps you should call the wrapper method :relaxSinglePoint");
	
	if(!lua_isnumber(L, 6))
		return luaL_error(L, "Number expected at for parameter 5");

	double step_size = lua_tonumber(L, 6);

	int num_steps = 1;
	if(lua_isnumber(L, 7))
		num_steps = lua_tointeger(L, 7);

	double goal2 = -1;
	if(lua_isnumber(L, 8))
	{
		goal2 = lua_tonumber(L, 8);
		goal2 *= goal2;
	}

	// need to save current configuration
	vector<double> cfg;
	saveConfiguration(L, get_index, cfg);

	double* vxyz = &state_xyz_path[p*num_sites*3];
	double* neighbour = new double[num_sites*3];
	

	// write path point as it currently stands
	setAllSpins(L, set_index, vxyz);

	int base = 7;
	if(currentSystem != Cartesian)
		base = 5;

	int* direction = new int[num_sites];
	double* grad = new double[num_sites*3];

	// seven_directions for each site (really 5 for sph/can)

	// find most relaxing direction
	const int maxd = pow(num_sites, base); // this many directions to consider
	double min_grad2 = 0;
	int steps_taken = 0;
	int direction_changes = 0;

	double* global_best_partials = new double[num_sites];
	double* previous_best_partials = new double[num_sites];
	double* current_partials = new double[num_sites];

	for(int i=0; i<num_sites; i++)
	{
		global_best_partials[i] = 1;
		previous_best_partials[i] = 1;
		current_partials[i] = 1;
	}
	
	while(num_steps > 0)
	{
		_D(FL, "num_steps", num_steps);
		int consecutive_fails = 0;
		int min_idx = 0;
		min_grad2 = 0;
		direction_changes++;
		
		memcpy(previous_best_partials, global_best_partials, sizeof(double) * num_sites);
		for(int i=0; i<num_sites; i++)
			previous_best_partials[i] = 1.0;

		// find most relaxing direction
		for(int i=0; i<maxd; i++)
		{
			if(_idx_2_number(i, base, direction, num_sites))
				break;
			
			for(int j=0; j<num_sites+1; j++) // +1 for "no change"
			{
				memcpy(current_partials, previous_best_partials, sizeof(double) * num_sites);
				bool first = true;
				bool making_progress = true;
				double last_g2 = 0;
				int pp = 5;
				while( (first || making_progress) && pp > 0)
				{
					pp--;
					if(j < num_sites)
					{
						current_partials[j] *= 1.5;
					}
					
					arrayCopyWithElementsChange(neighbour, vxyz, direction, step_size, current_partials, num_sites);
					computeVecFirstDerivative(L, neighbour, set_index, get_index, energy_index, grad);
				
					double g2 = dot(grad, grad, num_sites*3);
					
					if(i == 0 || g2 < min_grad2)
					{
						min_idx = i;
						min_grad2 = g2;
						_D(FL, "mg2", g2);
						memcpy(global_best_partials, current_partials, sizeof(double) * num_sites);
					}

					// local effort
					making_progress = (g2 < last_g2) || (first);
					first = false;
					last_g2 = g2;
				}
			}
		}
		
		if(min_idx) // then we've found a direction we can move with decreasing grads
		{
			_idx_2_number(min_idx, base, direction, num_sites);
			// this was calculated above, it's a good step
			arrayCopyWithElementsChange(vxyz, vxyz, direction, step_size, global_best_partials, num_sites); 

			_D(FL, "dir idx  ", min_idx);
			_D(FL, "direction", direction, num_sites);
			#if 1
			for(int qq=0; qq<num_sites; qq++)
			{
				_D(FL, "decode dir",  seven_directions[ direction[qq] ], 3);
			}
            #endif
			_D(FL, "good step", min_grad2);
			_D(FL, "search_mags", &(direction_search_magnitudes[0]), num_sites*3);

			grad_check(goal2, min_grad2, num_steps, end_reason_i);

			while(num_steps > 0 && consecutive_fails < relax_direction_fail_max)
			{ 
				_D(FL, "num_steps", num_steps);

				// keep trying to move in the direction
				arrayCopyWithElementsChange(neighbour, vxyz, direction, step_size, global_best_partials, num_sites);
				computeVecFirstDerivative(L, neighbour, set_index, get_index, energy_index, grad);
			
				double g2 = dot(grad, grad, num_sites*3);

				if(g2 < min_grad2)
				{
					arrayCopyWithElementsChange(vxyz, vxyz, direction, step_size, global_best_partials, num_sites); // make the move
					// extend scale in the direction
					step_size *= step_up;

					consecutive_fails = 0;
					min_grad2 = g2;
					_D(FL, "good step", g2);
					_D(FL, "search_mags", &(direction_search_magnitudes[0]), num_sites*3);
					
				}
				else
				{
					// reduce scale in the direction
					//if(step_size > 1e-80)
						step_size *= step_down;

					consecutive_fails++;
					_D(FL, "bad step", consecutive_fails);
				}
				num_steps--;
				steps_taken++;
				grad_check(goal2, min_grad2, num_steps, end_reason_i);
			}
		}
		else
		{
			// no luck on direction, retrying with smaller steps
			_D(FL, "no luck");
			//if(step_size > 1e-80)
				step_size *= step_down;

		}

		num_steps--;
		steps_taken++;
		grad_check(goal2, min_grad2, num_steps, end_reason_i);
	}
	steps_taken--;
	
	//	_D(FL, "vxyz2", vxyz, num_sites*3);
	
	// write updated cfg
 	setAllSpins(L, set_index, vxyz);

	delete [] neighbour;
	delete [] direction;
	delete [] grad;

	delete [] previous_best_partials;
	delete [] global_best_partials;
	delete [] current_partials;

	
	// need to restore saved configuration to SpinSystem
	loadConfiguration(L, set_index, cfg);

	lua_pushnumber(L, sqrt(min_grad2));
	lua_pushinteger(L, steps_taken);
	lua_pushinteger(L, direction_changes);

	lua_pushnumber(L, step_size);

	lua_pushstring(L, end_reason[end_reason_i]);

	return 5;
}
#endif


int MEP::relaxSinglePoint_expensiveDecent(lua_State* L, int get_index, int set_index, int energy_index, double* vxyz, double h, int steps)
{
	int good_steps = 0;
	const int num_sites = numberOfSites();

	double* grad   = new double[num_sites*3];
	double* vxyz2  = new double[num_sites*3];

	vector<double> mags;
	for(int i=0; i<num_sites; i++)
	{
		mags.push_back( _magnitude(&vxyz[i*3], currentSystem));
	}


	computeVecFirstDerivative(L, vxyz, set_index, get_index, energy_index, grad);
	double current_grad = sqrt(dot(grad, grad, num_sites*3));

	// printf("Start grad: %20e\n", current_grad);

	for(int i=0; i<steps; i++)
	{
		for(int qq=0; qq<num_sites*3; qq++)
		{
			memcpy(vxyz2, vxyz, sizeof(double)*num_sites*3);
			
			vxyz2[qq] = vxyz[qq] + h;

			for(int j=0; i<num_sites; i++)
			{
				_normalizeTo(&vxyz2[j*3+0], &vxyz2[j*3+0], currentSystem, mags[j]);
			}


			computeVecFirstDerivative(L, vxyz2, set_index, get_index, energy_index, grad);
			const double new_grad = sqrt(dot(grad, grad, num_sites*3));
			
			//printf("%20e <= %20e? %i\n", new_grad, current_grad,new_grad <= current_grad);
			if(new_grad <= current_grad)
			{
				//printf("$$$ %20e\n", new_grad);
				memcpy(vxyz, vxyz2, sizeof(double)*num_sites*3);
				current_grad = new_grad;
				good_steps++;
			}
		}
	}

	// printf("  end grad: %20e\n", current_grad);

	delete [] grad;
	delete [] vxyz2;
	energy_ok = false;
	return good_steps;
}


// relax individual point. This is used to refine a maximal point
// expected on the stack:
// at 1, mep
// at 2, point number
// at 3, function get_site_ss1(x,y,z) return sx,sy,sz 
// at 4, function set_site_ss1(x,y,z, sx,sy,sz)  return something_changed
// at 5, function get_energy_ss1()
// at 6, optional start h
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

	const int get_index = 3;
	const int set_index = 4;
	const int energy_index = 5;

	if(!lua_isfunction(L, 3) || !lua_isfunction(L, 4) || !lua_isfunction(L, 5))
		return luaL_error(L, "3 functions expected - perhaps you should call the wrapper method :relaxSinglePoint");
	
	// need to save current configuration
	vector<double> cfg;
	saveConfiguration(L, get_index, cfg);

	double* vxyz = &state_xyz_path[p*num_sites*3];

	// write path point as it currently stands
	setAllSpins(L, set_index, vxyz);

	double* grad   = new double[num_sites*3];
	double* vxyz2  = new double[num_sites*3];

	double* grad_grad = new double[num_sites*3];
	double h = epsilon;

	computeVecFirstDerivative(L, vxyz, set_index, get_index, energy_index, grad);
	_squashRadialComponent(currentSystem, grad, num_sites);
	double base_grad2 = dot(grad, grad, num_sites*3);

	for(int qq=0; qq<num_sites*3; qq++)
	{
		arrayCopyWithElementChange(currentSystem, vxyz2, vxyz, qq, -h, num_sites * 3);
		computeVecFirstDerivative(L, vxyz2, set_index, get_index, energy_index, grad);
		_squashRadialComponent(currentSystem, grad, num_sites);
		double x_minus_h = sqrt(dot(grad, grad, num_sites*3));

		arrayCopyWithElementChange(currentSystem, vxyz2, vxyz, qq, +h, num_sites * 3);
		computeVecFirstDerivative(L, vxyz2, set_index, get_index, energy_index, grad);
		_squashRadialComponent(currentSystem, grad, num_sites);
		double x_plus_h = sqrt(dot(grad, grad, num_sites*3));

		grad_grad[qq] = (x_plus_h - x_minus_h) / (2.0 * h);
	}

	_squashRadialComponent(currentSystem, grad_grad, num_sites);
#if 0
	for(int q=0; q<num_sites; q++)
	{
		double sf[3];
		_scaleFactors(currentSystem, &vxyz[q*3], sf);
		grad_grad[q*3+0] *= sf[0];
		grad_grad[q*3+1] *= sf[1];
		grad_grad[q*3+2] *= sf[2];
	}
#endif

	// gradient (of gradient magnitude) direction
	_normalizeTo(grad_grad, grad_grad, MEP::Cartesian, 1.0, num_sites*3);

#if 0
	printf("Direction:\n");
	for(int qq=0; qq<num_sites*3; qq++)
	{
		printf("%g ", grad_grad[qq]);
	}
	printf("\n");
#endif

	if(lua_isnumber(L, 6))
		h = lua_tonumber(L, 6);

	if(h < 1e-10)
		h = 1e-10;

	double goal = 0;
	if(lua_isnumber(L, 7))
	   goal = lua_tonumber(L, 7);



	// printf("(%s:%i) %g\n", __FILE__, __LINE__, sqrt(min2));

#if 1
	vector<double> mags;
	for(int i=0; i<num_sites; i++)
	{
		mags.push_back( _magnitude(&vxyz[i*3], currentSystem));
	}
#endif

	int max_steps = 50;
	int good_steps = 0;
	double min2 = base_grad2;
	const double start_min2 = min2;

	const double len_gg = sqrt(dot(grad_grad, grad_grad, num_sites*3));
	// printf(">>>>>>>>  h=%e  min2 > g2  (%e, %e)\n", h, min2, goal*goal);
	while(h > (epsilon * 1e-200) && max_steps && (min2 > goal*goal))
	{
		for(int i=0; i<num_sites; i++)
		{
			vxyz2[i*3+0] = vxyz[i*3+0] - grad_grad[i*3+0] * h;
			vxyz2[i*3+1] = vxyz[i*3+1] - grad_grad[i*3+1] * h;
			vxyz2[i*3+2] = vxyz[i*3+2] - grad_grad[i*3+2] * h;
			if(currentSystem == MEP::Cartesian)
				_normalizeTo(&vxyz2[i*3], &vxyz2[i*3], currentSystem, mags[i]);
		}

		computeVecFirstDerivative(L, vxyz2, set_index, get_index, energy_index, grad);
		_squashRadialComponent(currentSystem, grad, num_sites);
		double min2_2 = dot(grad, grad, num_sites*3);

		//printf("(%s:%i) %10e, %g\n", __FILE__, __LINE__, h, sqrt(min2_2));
		// printf("%10e %10e  %10e  %i\n", sqrt(min2_2), len_gg, h, min2_2 <= min2);
		if(min2_2 < min2)
		{
			consecutive_fails = 0;
			consecutive_successes++;

			min2 = min2_2;
			memcpy(vxyz, vxyz2, sizeof(double)*num_sites*3);
			for(int i=0; i<consecutive_successes; i++)
				h = h * step_up;
			good_steps++;
			//printf("+ %e\n", h);
			#if 0
			printf("good step: %e -> %e      h = %e\n", min2, min2_2, h);
			for(int i=0; i<num_sites*3; i++)
				printf("%g ", vxyz[i]);
			printf("\n");
			#endif
		}
		else
		{
			consecutive_fails++;
			consecutive_successes = 0;

			for(int i=0; i<consecutive_fails; i++)
				h = h * step_down;
			//printf("- %e\n", h);
			// printf(" bad step: %e -> %e      h = %e\n", min2, min2_2, h);
		}
		max_steps--;
	}

#if 0
	if(min2 > goal*goal)
		if(start_min2 > 0)
			if(min2/start_min2 > 0.9999999 ) // then stalling
	{
		// printf("STALLING\n");
		int good_steps = relaxSinglePoint_expensiveDecent(L, get_index, set_index, energy_index, vxyz, 1e-10, 10);
	}
#endif

	// write updated cfg
 	// setAllSpins(L, set_index, vxyz);

	delete [] grad;
	delete [] grad_grad;
	delete [] vxyz2;

	//need to restore saved configuration to SpinSystem
	loadConfiguration(L, set_index, cfg);

	lua_pushnumber(L, sqrt(min2));
	lua_pushnumber(L, h);
	lua_pushinteger(L, good_steps);

	energy_ok = false;
	return 3;
}



// relax individual point. This is used to refine a maximal point
// expected on the stack:
// at 1, mep
// at 2, function get_site_ss1(x,y,z) return sx,sy,sz 
// at 3, function set_site_ss1(x,y,z, sx,sy,sz)  return something_changed
// at 4, function get_energy_ss1()
// at 5, point number
// at 6, direction
// at 7, num steps
// at 8, step size
// at 9, tolerance
// returns new step size
int MEP::slidePoint(lua_State* L)
{
	const int num_sites = numberOfSites();
	const int path_length = numberOfImages();

	const int get_index = 2;
	const int set_index = 3;
	const int energy_index = 4;

	const double step_up = 1.1;
	const double step_down = 0.9;

	const int p = lua_tointeger(L, 5) - 1;
	if(p < 0 || p >= path_length)
		return luaL_error(L, "Point index out of bounds");

	double direction = lua_tonumber(L, 6);
	if(direction < 0)
		direction = -1;
	else
		direction =  1;

	int num_steps = lua_tointeger(L, 7);
	double h = lua_tonumber(L, 8);

	if(h == 0)
		h = 1e-10;

	// back up old sites so we can restore after
	vector<double> cfg;
	saveConfiguration(L, get_index, cfg);

	const int n = num_sites*3;
	double* vec = &state_xyz_path[p*n];

	double* slope = new double[n];
	double* state = new double[n];

	double* mags = new double[num_sites];
	for(int i=0; i<num_sites; i++)
	{
		mags[i] = _magnitude(currentSystem, &(vec[i*3]));
	}

	
	// write path point as it currently stands (at this point)
	setAllSpins(L, set_index, vec);
	double last_energy = getEnergy(L, energy_index);
	int consecutive_fails = 0;
	int consecutive_successes = 0;


	for(int i=0; i<num_steps; i++)
	{
		computeVecFirstDerivative(L, vec, set_index, get_index, energy_index, slope);

		for(int q=0; q<num_sites; q++)
		{
			double sf[3];
			_scaleFactors(currentSystem, &vec[q*3], sf);
			slope[q*3+0] *= sf[0];
			slope[q*3+1] *= sf[1];
			slope[q*3+2] *= sf[2];
		}

		// gradient  direction
		// the following line looks odd. The idea is to take the
		// list of numbers representing the gradient of all spins
		// and scale it down so the length, if it were treated as an
		// n-dimensional cartesian vector, is one.  
		_normalizeTo(slope, slope, MEP::Cartesian, 1.0, n);

		for(int j=0; j<n; j++)
		{
			state[j] = vec[j] + h * slope[j] * direction;
		}

		rescale_vectors(currentSystem, state, mags, num_sites);
		setAllSpins(L, set_index, state);
		double current_energy = getEnergy(L, energy_index);

		// printf("%g %e %e    %e\n", direction, current_energy, last_energy, h);

		if(direction < 0)
		{
			if(current_energy < last_energy)
			{
				// printf("(%s:%i)\n", __FILE__, __LINE__);
				consecutive_successes++;
				consecutive_fails = 0;
				for(int j=0; j<consecutive_successes; j++)
					h *= step_up;
				memcpy(vec, state, sizeof(double)*n);
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
				memcpy(vec, state, sizeof(double)*n);
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

	loadConfiguration(L, set_index, cfg);	
	energy_ok = false;
	delete [] slope;
	delete [] mags;
	delete [] state;

	lua_pushnumber(L, last_energy);
	return 1;
}



int MEP::numberOfImages()
{
	if(sites.size() == 0)
		return 0;
	return state_xyz_path.size() / sites.size();
}

int MEP::numberOfSites()
{
	return sites.size() / 3;
}


bool MEP::equal(int a, int b, double allowable)
{
	const int chunk = sites.size();

	double* s1 = &state_xyz_path[a * chunk];
	double* s2 = &state_xyz_path[b * chunk];

	for(int i=0; i<numberOfSites(); i++)
	{
		if(_angleBetween(&s1[i*3], &s2[i*3], currentSystem, currentSystem) > allowable)
			return false;
	}

	return true;
}

#include <set>
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
			if(equal( consider_sites[i], unique_sites[j], tol))
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

void MEP::computePointGradAtSite(lua_State* L, int p, int s, int set_index, int energy_index, double* grad3)
{
	double* vec = &state_xyz_path[p*numberOfSites()*3 + s*3];
	
	const double m1 = _magnitude(currentSystem, vec);
	
	double h[3];
	_stepSize(currentSystem, vec, epsilon, h);

	// We now have a point at site p,s and appropriate step sizes
	// time to compute the spatial energy gradient.
	double g[6] = {0,0,0,0,0,0};

	// 7th below is to restore old state
	const double cc[7][3] = {{1,0,0},{-1,0,0},   {0,1,0},{0,-1,0},   {0,0,1},{0,0,-1},   {0,0,0}};
	for(int c=0; c<7; c++)
	{
		double new_coord[3];
		for(int i=0; i<3; i++)
			new_coord[i] = vec[i] + cc[c][i] * h[i];

		_normalizeTo(new_coord, new_coord, currentSystem, m1);

		setSiteSpin(L, set_index, &sites[s*3], new_coord);
		
		if(c < 6) // last iteration resets cfg. No need to calc energy in that case.
		{
			g[c] = getEnergy(L, energy_index);
		}
	}
		
// 	printf("(%s:%i) %e %e\n", __FILE__, __LINE__, vxyz[0], grad3[0]);

	double sf[3];
	_scaleFactors(currentSystem, vec, sf);
		
	grad3[0] = sf[0] * (g[1] - g[0]) / (2.0 * h[0]);
	grad3[1] = sf[1] * (g[3] - g[2]) / (2.0 * h[1]);
	grad3[2] = sf[2] * (g[5] - g[4]) / (2.0 * h[2]);
	
/*
 	printf("h3: %e %e %e\n", h[0], h[1], h[2]);
	printf("sf: %e %e %e\n", sf[0], sf[1], sf[2]);
	printf("gr: %e %e %e\n", g[1] - g[0], g[3] - g[2], g[5] - g[4]);
	printf("%e %e %e --> %e %e %e\n", vec[0], vec[1], vec[2], grad3[0], grad3[1], grad3[2]);
*/
}

void MEP::writePathPoint(lua_State* L, int set_index, double* vxyz)
{
	const int num_sites = numberOfSites();
	
	// write path point as it is given
	for(int s=0; s<num_sites; s++)
	{
		setSiteSpin(L, set_index, &sites[s*3], &vxyz[s*3]);
	}
	
// 	energy_ok = false;
// 	good_distances = false;
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
	const int path_length = numberOfImages();
	const int ns = numberOfSites();

	if(!lua_isnumber(L, 2) || !lua_isnumber(L, 3))
		return luaL_error(L, "Require 2 numbers");

	int a = lua_tointeger(L, 2) - 1;
	int b = lua_tointeger(L, 3) - 1;
	
	if( (a < 0) || (b < 0) || (a >= path_length) || (b >= path_length))
		return luaL_error(L, "Require 2 numbers between 1 and number of points (%i)", path_length);

   	const int chunk = sites.size();

	double* s1 = &state_xyz_path[a * chunk];
	double* s2 = &state_xyz_path[b * chunk];

	   
	lua_newtable(L);
  	for(int i=0; i<ns; i++)
	{
		lua_pushinteger(L, i+1);

		double angle = _angleBetween(&s1[i*3], &s2[i*3], currentSystem, currentSystem);
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
		return luaL_error(L, "Energies not found. You may need to run a single round of :compute");

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



int MEP::calculateEnergyGradients(lua_State* L, int get_index, int set_index, int energy_index)
{
	const int ns = numberOfSites();
	const int ni = numberOfImages();
	
	force_vector.resize( state_xyz_path.size() ); //since we're doing random-ish access

	double sf[3] = {1,1,1};
	
	// lets march along the path
	for(int i=0; i<ni; i++)
	{
		double* vec = &state_xyz_path[i*ns*3];
		double* force = &force_vector[i*ns*3];
		computeVecFirstDerivative(L, vec, set_index, get_index, energy_index, force);


		
		for(int j=0; j<ns; j++)
		{
			//_scaleFactors(currentSystem, &vec[j*3], sf);
			force[j*3+0] *= sf[0];
			force[j*3+1] *= sf[1];
			force[j*3+2] *= sf[2];
		}
		//printf("sf %e %e %e\n", sf[0], sf[1], sf[2]);
	}
	
	for(int i=0; i<state_xyz_path.size(); i++)
	{
		force_vector[i] *= beta;
		// printf("fv[%i] = %e\n", i, force_vector[i]);
	}

	return 0;
}

int MEP::classifyPoint(lua_State* L)
{
	const int num_sites = numberOfSites();
	const int path_length = numberOfImages();

	const int get_index = 2;
	const int set_index = 3;
	const int energy_index = 4;

	const int p = lua_tointeger(L, 5) - 1;
	if(p < 0 || p >= path_length)
		return luaL_error(L, "Point index out of bounds");


	double h = epsilon;
	if(lua_isnumber(L, 6))
		h = lua_tonumber(L, 6);

	// back up old sites so we can restore after
	vector<double> cfg;
	saveConfiguration(L, get_index, cfg);

	double* vec = &state_xyz_path[p*num_sites*3];
	double* state = new double[num_sites * 3];
	double* mags = new double[num_sites];

	for(int i=0; i<num_sites; i++)
	{
		mags[i] = _magnitude(currentSystem, &(vec[i*3]));
	}

	arrayCopyWithElementChange(currentSystem, state, vec, 0, 0, num_sites * 3);
	rescale_vectors(currentSystem, state, mags, num_sites);
	setAllSpins(L, set_index, state);
	const double base_energy = getEnergy(L, energy_index);
	

	int ups = 0;
	int downs = 0;
	int equals = 0;
	
	for(int i=0; i<num_sites*3; i++)
	{
		arrayCopyWithElementChange(currentSystem, state,   vec,  i, -h, num_sites * 3);
		rescale_vectors(currentSystem, state, mags, num_sites);
		setAllSpins(L, set_index, state);
		const double e1 = getEnergy(L, energy_index);

		arrayCopyWithElementChange(currentSystem, state,   vec,  i,  h, num_sites * 3);
		rescale_vectors(currentSystem, state, mags, num_sites);
		setAllSpins(L, set_index, state);
		const double e2 = getEnergy(L, energy_index);
		
		if(e1 == base_energy && e1 == e2)
			equals++;

		if(e1 > base_energy && e2 > base_energy)
		{
			ups++;
		}
		if(e1 < base_energy && e2 < base_energy)
		{
			downs++;
		}
	}

	loadConfiguration(L, set_index, cfg);	

	delete [] state;	
	delete [] mags;
	
	if(ups && (downs == 0))
	{
		lua_pushstring(L, "Minimum");
		return 1;
	}
	if(downs && (ups == 0))
	{
		lua_pushstring(L, "Maximum");
		return 1;
	}
	lua_pushstring(L, "");
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
	mep->state_xyz_path.clear();
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
	
	double xyz[3];

	xyz[0] = lua_tonumber(L, 2);
	xyz[1] = lua_tonumber(L, 3);
	xyz[2] = lua_tonumber(L, 4);
	
	double coord[3];

	_convertCoordinateSystem(MEP::Cartesian, mep->currentSystem, xyz, coord);

	mep->state_xyz_path.push_back(coord[0]);
	mep->state_xyz_path.push_back(coord[1]);
	mep->state_xyz_path.push_back(coord[2]);
	
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


static int l_resampleStateXYZPath(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	return mep->resampleStateXYZPath(L);
}

static int l_projForcePerpSpins(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	
	int get_index = 2;
	int set_index = 3;
	int energy_index = 4;
	
	if(!lua_isfunction(L, get_index))
		return luaL_error(L, "1st argument expected to be get_site_ss1(x,y,z) function");
	if(!lua_isfunction(L, set_index))
		return luaL_error(L, "2nd argument expected to be set_site_ss1(x,y,z,sx,sy,sz) function");
	if(!lua_isfunction(L, energy_index))
		return luaL_error(L, "3rd argument expected to be get_energy_ss1() function");
	
	mep->projForcePerpSpins(L, get_index, set_index, energy_index);
	return 0;
}
static int l_projForcePerpPath(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	
	int get_index = 2;
	int set_index = 3;
	int energy_index = 4;
	
	if(!lua_isfunction(L, get_index))
		return luaL_error(L, "1st argument expected to be get_site_ss1(x,y,z) function");
	if(!lua_isfunction(L, set_index))
		return luaL_error(L, "2nd argument expected to be set_site_ss1(x,y,z,sx,sy,sz) function");
	if(!lua_isfunction(L, energy_index))
		return luaL_error(L, "3rd argument expected to be get_energy_ss1() function");
	mep->projForcePerpPath(L, get_index, set_index, energy_index);
	return 0;
}
static int l_projForcePath(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	
	int get_index = 2;
	int set_index = 3;
	int energy_index = 4;
	
	if(!lua_isfunction(L, get_index))
		return luaL_error(L, "1st argument expected to be get_site_ss1(x,y,z) function");
	if(!lua_isfunction(L, set_index))
		return luaL_error(L, "2nd argument expected to be set_site_ss1(x,y,z,sx,sy,sz) function");
	if(!lua_isfunction(L, energy_index))
		return luaL_error(L, "3rd argument expected to be get_energy_ss1() function");
	
	mep->projForcePath(L, get_index, set_index, energy_index);
	return 0;
}

// at 1, mep
// at 2, function get_site(x,y,z) return sx,sy,sz end
// at 3, function set_site(x,y,z, sx,sy,sz) end
// at 4, get energy function
static int l_calculateEnergyGradients(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	
	int get_index = 2;
	int set_index = 3;
	int energy_index = 4;
	
	if(!lua_isfunction(L, get_index))
		return luaL_error(L, "1st argument expected to be get_site_ss1(x,y,z) function");
	if(!lua_isfunction(L, set_index))
		return luaL_error(L, "2nd argument expected to be set_site_ss1(x,y,z,sx,sy,sz) function");
	if(!lua_isfunction(L, energy_index))
		return luaL_error(L, "3rd argument expected to be get_energy_ss1() function");
	
	return mep->calculateEnergyGradients(L, get_index, set_index, energy_index);
}

// at 1, mep
// at 2, function get_site(x,y,z) return sx,sy,sz end
// at 3, function set_site(x,y,z, sx,sy,sz) end
// at 4, get energy function
static int l_calculateEnergies(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	
	int get_index = 2;
	int set_index = 3;
	int energy_index = 4;
	
	if(!lua_isfunction(L, get_index))
		return luaL_error(L, "1st argument expected to be get_site_ss1(x,y,z) function");
	if(!lua_isfunction(L, set_index))
		return luaL_error(L, "2nd argument expected to be set_site_ss1(x,y,z,sx,sy,sz) function");
	if(!lua_isfunction(L, energy_index))
		return luaL_error(L, "3rd argument expected to be get_energy_ss1() function");
	
	return mep->calculateEnergies(L, get_index, set_index, energy_index);
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

	const int p = lua_tointeger(L, 2) - 1;
	const int s = lua_tointeger(L, 3) - 1;

	if(p < 0 || s < 0)
		return 0;

	int idx = mep->sites.size() * p + s*3;

	if(mep->force_vector.size() < idx+2)
		return 0;

	lua_pushnumber(L, mep->force_vector[idx+0]);
	lua_pushnumber(L, mep->force_vector[idx+1]);
	lua_pushnumber(L, mep->force_vector[idx+2]);
	return 3;
}

// expected on the stack at function call
// at 1, mep
// at 2, point number
// at 3, function get_site_ss1(x,y,z) return sx,sy,sz 
// at 4, function set_site_ss1(x,y,z, sx,sy,sz) 
// at 5, function get_energy_ss1()
// this is the internal version
static int l_relaxSinglePoint_sd_(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	return mep->relaxSinglePoint_SteepestDecent(L);
}


static int l_slidePoint_(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);
	return mep->slidePoint(L);
}


// expected on the stack at function call
// at 1, mep
// at 2, point number
// at 3, function get_site_ss1(x,y,z) return sx,sy,sz 
// at 4, function set_site_ss1(x,y,z, sx,sy,sz) 
// at 5, function get_energy_ss1()
// this is the internal version
static int l_computepoint2deriv(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	const int get_index = 3;
	const int set_index = 4;
	const int energy_index = 5;
	
	int num_sites = mep->numberOfSites();
	
	int p = lua_tointeger(L, 2) - 1;
	
	if(p < 0 || p >= mep->numberOfImages())
		return luaL_error(L, "Invalid point number");

	double* derivsAB = new double[num_sites * num_sites * 9];
	
	mep->computePointSecondDerivative(L, p, set_index, get_index, energy_index, derivsAB);
	
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
// at 3, function get_site_ss1(x,y,z) return sx,sy,sz 
// at 4, function set_site_ss1(x,y,z, sx,sy,sz) 
// at 5, function get_energy_ss1()
// this is the internal version
static int l_computepoint1deriv(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	const int get_index = 3;
	const int set_index = 4;
	const int energy_index = 5;
	
	int num_sites = mep->numberOfSites();
	
	int p = lua_tointeger(L, 2) - 1;
	
	if(p < 0 || p >= mep->numberOfImages())
		return luaL_error(L, "Invalid point number");

	double* derivs = new double[num_sites * 3];
	
	mep->computePointFirstDerivative(L, p, set_index, get_index, energy_index, derivs);
	
	lua_newtable(L);
	for(int i=0; i<num_sites * num_sites * 9; i++)
	{
		lua_pushinteger(L, i+1);
		lua_pushnumber(L, derivs[i]);
		lua_settable(L, -3);
	}
	delete [] derivs;
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

	if(p < 0 || s < 0)
	{
		return luaL_error(L, "Path Point or site is out of bounds. {Point,Site} = {%d,%d}. Upper Bound = {%d,%d}", p+1,s+1,mep->numberOfImages(), mep->numberOfSites());
	}

	int idx = mep->sites.size() * p + s*3;

	if(mep->state_xyz_path.size() < idx+2)
	{
		return luaL_error(L, "Path Point or site is out of bounds. {Point,Site} = {%d,%d}. Upper Bound = {%d,%d}", p+1,s+1,mep->numberOfImages(), mep->numberOfSites());
	}
	
	double x[3];
	double c[3];

	x[0] = mep->state_xyz_path[idx+0];
	x[1] = mep->state_xyz_path[idx+1];
	x[2] = mep->state_xyz_path[idx+2];
	const double m = _magnitude(mep->currentSystem, x);
	
	_convertCoordinateSystem(mep->currentSystem, MEP::Cartesian, x, c);
	
	lua_pushnumber(L, c[0]);
	lua_pushnumber(L, c[1]);
	lua_pushnumber(L, c[2]);
	lua_pushnumber(L, m);
	return 4;
}

	
static int l_setsite(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	const int p = lua_tointeger(L, 2) - 1;
	const int s = lua_tointeger(L, 3) - 1;

	if(p < 0 || s < 0)
	{
		return luaL_error(L, "Path Point or site is out of bounds. {Point,Site} = {%d,%d}. Upper Bound = {%d,%d}", p+1,s+1,mep->numberOfImages(), mep->numberOfSites());
	}

	int idx = mep->sites.size() * p + s*3;

	if(mep->state_xyz_path.size() < idx+2)
	{
		return luaL_error(L, "Path Point or site is out of bounds. {Point,Site} = {%d,%d}. Upper Bound = {%d,%d}", p+1,s+1,mep->numberOfImages(), mep->numberOfSites());
	}

	double x,y,z;
	
	if(lua_istable(L, 4))
	{
		lua_pushinteger(L, 1);
		lua_gettable(L, 4);
		x = lua_tonumber(L, -1);
		lua_pop(L, 1);

		lua_pushinteger(L, 2);
		lua_gettable(L, 4);
		y = lua_tonumber(L, -1);
		lua_pop(L, 1);

		lua_pushinteger(L, 3);
		lua_gettable(L, 4);
		z = lua_tonumber(L, -1);
		lua_pop(L, 1);
	}
	else
	{
		x = lua_tonumber(L, 4);
		y = lua_tonumber(L, 5);
		z = lua_tonumber(L, 6);
	}
	
	double v[3];
	double c[3];

	v[0] = x;
	v[1] = y;
	v[2] = z;


	_convertCoordinateSystem(MEP::Cartesian, mep->currentSystem, v, c);

	mep->state_xyz_path[idx+0] = c[0];
	mep->state_xyz_path[idx+1] = c[1];
	mep->state_xyz_path[idx+2] = c[2];

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
	const double d = mep1->absoluteDifference(mep2, point, max_diff, max_idx);

	if(d == -1)
		return luaL_error(L, "internal state size mismatch");

	lua_pushnumber(L, d);
	lua_pushnumber(L, max_diff);
	lua_pushinteger(L, max_idx+1);
	return 3;
}

static int l_debug(lua_State* L)
{
	LUA_PREAMBLE(MEP, mep, 1);

	lua_pushinteger(L, mep->state_xyz_path.size());
	return 1;
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

	const int get_index = 3;
	const int set_index = 4;
	const int energy_index = 5;
	
	vector<double> nderiv;

	mep->calculatePathEnergyNDeriv(L, get_index, set_index, energy_index, n, nderiv);
	
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

static int l_conv_cs(lua_State* L)
{
	double src[3];
	double dest[3];

	LUA_PREAMBLE(MEP, mep, 1);
	MEP::CoordinateSystem src_cs;
	MEP::CoordinateSystem dest_cs;

	src[0] = lua_tonumber(L, 2);
	src[1] = lua_tonumber(L, 3);
	src[2] = lua_tonumber(L, 4);

	src_cs = (MEP::CoordinateSystem) lua_tointeger(L, 5);
	dest_cs = (MEP::CoordinateSystem) lua_tointeger(L, 6);
	
	_convertCoordinateSystem(src_cs, dest_cs, src, dest);

	lua_pushnumber(L, dest[0]);
	lua_pushnumber(L, dest[1]);
	lua_pushnumber(L, dest[2]);

	return 3;
}


static int l_computePointSecondDerivativeAB(lua_State* L)
{
    LUA_PREAMBLE(MEP, mep, 1);

	const int pt = lua_tonumber(L, 2) - 1;
	const int c1 = lua_tonumber(L, 3) - 1;
	const int c2 = lua_tonumber(L, 4) - 1;

	const int get_index = 5;
    const int set_index = 6;
    const int energy_index = 7;

	double d1 = -1;
	double d2 = -1;

	if(lua_isnumber(L, 8))
		d1 = lua_tonumber(L, 8);
	if(lua_isnumber(L, 9))
		d2 = lua_tonumber(L, 9);


	lua_pushnumber(L, mep->computePointSecondDerivativeAB(L, pt, set_index, get_index, energy_index, c1, c2, d1, d2));

	return 1;
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
		lua_pushstring(L, "4 Numbers: x,y,z,m orientation of spin and magnitude at site s at path point p.");
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
	if(func == l_ppc)
	{
		lua_pushstring(L, "Get number of path points that currently exist in the object.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 Integer: number of path points.");
		return 3;
	}
	
	if(func == l_absoluteDifference)
	{
		lua_pushstring(L, "Get the absolute difference between the calling MEP and the passed MEP. Difference between two moments m1 and m2 is defined as the norm of the vector D where D = normalized(m1) - normalized(m2).");
		lua_pushstring(L, "1 MEP: MEP to compare against.");
		lua_pushstring(L, "2 Numbers, 1 Integer: Total of all differences, the maximum single difference and the index where that max occurs.");
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
		lua_pushstring(L, "Set the internal coordinate system. This can be changed during the calculation.");
		lua_pushstring(L, "1 String: Must match one of the values in the table returned by :coordinateSystems.");
		lua_pushstring(L, "");
		return 3;
	}

	if(func == _l_getCoordinateSystem)
	{
		lua_pushstring(L, "Get the internal coordinate system.");
		lua_pushstring(L, "");
		lua_pushstring(L, "1 String: Matches one of the values in the table returned by :coordinateSystems.");
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

	if(func == l_us)
	{
		lua_pushstring(L, "Get a list of which sites that are unique.");
		lua_pushstring(L, "Zero or 1 number, 0 or 1 table: tolerances used to determine equality. If a number is provided it will be used. The tolerance is the fraction of the spin magnitude that two vectors can be different by. If a table is provided then only the points in the table will be considered.");
		lua_pushstring(L, "1 Table: Indexes of unique sites");
		return 3;
	}


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
		{"gradient", l_getgradient},
		{"spin", l_getsite},
		// {"spinInCoordinateSystem", l_getsite},
		{"setSpin", l_setsite},
		// {"setSpinInCoordinateSystem", l_setsite},
		{"applyForces", l_applyforces},
		{"_addStateXYZ", l_addstatexyz},
		{"_setImageSiteMobility", l_setimagesitemobility},
		{"_relaxSinglePoint_sd", l_relaxSinglePoint_sd_}, //internal method
		{"_hessianAtPoint", l_computepoint2deriv},
		{"_gradAtPoint", l_computepoint1deriv},
		{"_maximalPoints", l_maxpoints},
		{"_randomize", l_randomize_},
		{"internalCopyTo", l_internal_copyto},
		{"absoluteDifference", l_absoluteDifference},
		{"setBeta", l_setbeta},
		{"beta", l_getbeta},
		{"_debug", l_debug},
		{"_pathEnergyNDeriv", l_pend_},
		{"coordinateSystems", _l_getCoordinateSystems},
		{"coordinateSystem", _l_getCoordinateSystem},
		{"setCoordinateSystem", _l_setCoordinateSystem},
		{"_convertCoordinateSystem", l_conv_cs},
		{"epsilon", l_getep},
		{"setEpsilon", l_setep},
		{"uniqueSites", l_us},
		{"setRelaxDirectionFailMax", l_setrfm},
		{"relaxDirectionFailMax", l_getrfm},
		{"_slidePoint", l_slidePoint_},
		{"_classifyPoint", l_cp_},
		{"anglesBetweenPoints", l_asbs},
		{"_computePointSecondDerivativeAB", l_computePointSecondDerivativeAB},
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
	if(luaL_dostring(L, __mep_luafuncs()))
	{
		fprintf(stderr, "%s\n", lua_tostring(L, -1));
		return luaL_error(L, lua_tostring(L, -1));
	}

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
