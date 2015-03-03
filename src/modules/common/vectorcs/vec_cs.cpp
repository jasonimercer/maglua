#include "vec_cs.h"

#include <stdio.h> //fprintf
#include <string.h> //memcpy
#include <strings.h> // strncasecmp
#include <math.h>
#ifndef M_PI
#define M_PI 3.141592653
#endif

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


#define _acos(x) _acosFL(x, __FILE__, __LINE__)

static double _acosFL(double x, const char* file, const int line)
{
    if(fabs(x) > 1)
    {
        //{int* i = (int*)5; *i = 5;}
        //fprintf(stderr, "(%s:%i) argument of acos out of range (%g)\n", file, line, x);
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

double VectorCS::magnitude() const
{
	if(cs == Cartesian)
		return sqrt(dot(v,v));
	return v[0];
}


VectorCS& VectorCS::setMagnitude(double mag)
{
	if(cs == Cartesian)
	{
		double m = magnitude();
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
	return *this;
}

static void _squashRadialComponent(CoordinateSystem cs, double* vec)
{
	if(cs == Cartesian)
		return;
	vec[0] = 0;
}


void VectorCS::zeroRadialComponent()
{
	if(cs == Undefined || cs == Cartesian)
		return;
	v[0] = 0;
}

VectorCS VectorCS::zeroedRadialComponent() const
{
	VectorCS vec(*this);
 	vec.zeroRadialComponent();
	return vec;
}


void VectorCS::zeroRadialComponent(const VectorCS& radial)
{
	if(cs == Cartesian)
	{
		reject(radial);
	}
	else
	{
		return zeroRadialComponent();
	}
}

VectorCS VectorCS::zeroedRadialComponent(const VectorCS& radial) const
{
	VectorCS vec(*this);
 	vec.zeroRadialComponent(radial);
	return vec;
}





// the next few functions deal with rotating coordinate systems
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

static char csRotatedType(CoordinateSystem cs)
{
	switch(cs)
	{
	case Undefined:
	case Cartesian:
	case Spherical:
	case Canonical:
		return ' ';
	case SphericalX:
	case CanonicalX:
		return 'X';
	case SphericalY:
	case CanonicalY:
		return 'Y';
	}
	return ' ';
}

static CoordinateSystem csRotate(CoordinateSystem cs, char r)
{
	if(cs == Spherical)
	{
		switch(r)
		{
		case 'x':
		case 'X':
			return SphericalX;
		case 'y':
		case 'Y':
			return SphericalY;
		}
	}
	if(cs == Canonical)
	{
		switch(r)
		{
		case 'x':
		case 'X':
			return CanonicalX;
		case 'y':
		case 'Y':
			return CanonicalY;
		}
	}
	return Undefined;
}

static CoordinateSystem csUnrotated(CoordinateSystem cs)
{
	switch(cs)
	{
	case Undefined:
	case Cartesian:
	case Spherical:
	case Canonical:
		return cs;
 	case SphericalX:
	case SphericalY:
		return Spherical;
	case CanonicalX:
	case CanonicalY:
		return Canonical;
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

static void _convertCoordinateSystem(
	CoordinateSystem src_cs, CoordinateSystem dest_cs,
	const double* src, double* dest)
{
	if(src_cs == dest_cs)
	{
		memcpy(dest, src, 3 * sizeof(double));
		return;
	}

	if(src_cs == Undefined || dest_cs == Undefined)
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
		_convertCoordinateSystem(src_cs, src_cs, src, temp);
		src = temp; // to make sure we're not stomping around like morons, overwriting parts of src with dest
	}

	char src_r = csRotatedType(src_cs);
	if(src_r != ' ') //then it's a rotated type
	{
		_convertCoordinateSystem(csUnrotated(src_cs), Cartesian, src, dest);
		rotate(dest, src_r, -1); //remove rotation type
		_convertCoordinateSystem(Cartesian, dest_cs, dest, dest);
	}

	char dest_r = csRotatedType(dest_cs);
	if(dest_r != ' ') // then it's a rotated type
	{
		if(src_cs == Cartesian)
		{
			// do the rotation
			memcpy(dest, src, sizeof(double)*3);
			rotate(dest, dest_r, 1); // apply rotation type
			_convertCoordinateSystem(src_cs, csUnrotated(dest_cs), dest, dest);
			return;
		}
		else
		{
			double c[3];
			_convertCoordinateSystem(src_cs, Cartesian, src, c);
			_convertCoordinateSystem(Cartesian, dest_cs, c, dest);
			return;
		}
	}

	if(src_cs == Cartesian)
	{
		const double x = src[CARTESIAN_X];
		const double y = src[CARTESIAN_Y];
		const double z = src[CARTESIAN_Z];

		const double r = sqrt(x*x + y*y + z*z);
		const double p = atan2(y, x);
		const double t = _acos(z / r);

		if(dest_cs == Spherical)
		{
			dest[SPHERICAL_R] = r;
			dest[SPHERICAL_PHI] = p;
			if(r == 0)
				dest[SPHERICAL_THETA] = 0;
			else
				dest[SPHERICAL_THETA] = t;
		}
		if(dest_cs == Canonical)
		{
			dest[CANONICAL_R] = r;
			dest[CANONICAL_PHI] = p;
			if(r == 0)
				dest[CANONICAL_P] = 1;
			else
				dest[CANONICAL_P] = cos(t);
		}
	}
	if(src_cs == Spherical)
	{
		if(dest_cs == Cartesian)
		{
		    double r = src[SPHERICAL_R];
		    double p = src[SPHERICAL_PHI];
		    double t = src[SPHERICAL_THETA];

		    dest[CARTESIAN_X] = src[0] * cos( p ) * sin( t );
		    dest[CARTESIAN_Y] = src[0] * sin( p ) * sin( t );
		    dest[CARTESIAN_Z] = src[0] * cos( t );
		}
		else
		{
		    double t[3];
		    _convertCoordinateSystem(src_cs, Cartesian, src, t);
		    _convertCoordinateSystem(Cartesian, dest_cs, t, dest);
		}

	}
	if(src_cs == Canonical)
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

		if(dest_cs == Cartesian)
		{
			const double t = _acos(p);
			dest[CARTESIAN_X] = r * cos( phi ) * sin( t );
			dest[CARTESIAN_Y] = r * sin( phi ) * sin( t );
			dest[CARTESIAN_Z] = r * p;
		}
		else
		{
		    double t[3];
		    _convertCoordinateSystem(src_cs, Cartesian, src, t);
		    _convertCoordinateSystem(Cartesian, dest_cs, t, dest);
		}
	}
}

void VectorCS::convertToCoordinateSystem(CoordinateSystem new_cs)
{
	double dest[3];
	_convertCoordinateSystem(cs, new_cs, v, dest);
	memcpy(v, dest, sizeof(double)*3);
	cs = new_cs;
}

VectorCS VectorCS::convertedToCoordinateSystem(CoordinateSystem new_cs) const
{
	VectorCS vv = copy();
	vv.convertToCoordinateSystem(new_cs);
	return vv;
}

VectorCS VectorCS::normalizedTo(double length) const
{
	return copy().setMagnitude(length);
}


// bring the values for periodic coordinates in good looking ranges
void VectorCS::fix()
{
    return;
    if(cs == Cartesian || cs == Undefined)
	return; // nothing to be done here

    if(cs == Canonical)
    {
	double& phi   = v[1];
	double& p     = v[2];


	// done thinking about the "smart" way to do this:
	if(p < -1 || p > 1 || phi < 0 || phi > 2*M_PI)
	{
	    convertToCoordinateSystem(Cartesian);
	    convertToCoordinateSystem(Canonical);
	}
    }
    
    if(cs == Spherical)
    {
	double& phi   = v[1];
	double& theta = v[2];


	// done thinking about the "smart" way to do this:
	if(theta < 0 || theta > M_PI || phi < 0 || phi > 2*M_PI)
	{
	    convertToCoordinateSystem(Cartesian);
	    convertToCoordinateSystem(Spherical);
	}
    }

}

void VectorCS::scaleFactors(double* sf)
{
	const double* vec = v;

	switch(cs)
	{
	case Cartesian:
		sf[CARTESIAN_X] = 1;
		sf[CARTESIAN_Y] = 1;
		sf[CARTESIAN_Z] = 1;
		break;
	case Spherical:
	case SphericalX:
	case SphericalY:
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
	case Canonical:
	case CanonicalX:
	case CanonicalY:
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

void VectorCS::stepSize(const double epsilon, double* h3)
{
	const double m = magnitude();
	switch(cs)
	{
	case Cartesian:
		h3[CARTESIAN_X] = m*epsilon;
		h3[CARTESIAN_Y] = m*epsilon;
		h3[CARTESIAN_Z] = m*epsilon;
		break;
	case Spherical:
	case SphericalX:
	case SphericalY:
		h3[SPHERICAL_R] = m*epsilon;
		h3[SPHERICAL_PHI] = 2*M_PI*epsilon;
		h3[SPHERICAL_THETA] = M_PI*epsilon;
		break;
	case Canonical:
	case CanonicalX:
	case CanonicalY:
		h3[CANONICAL_R] = m*epsilon;
		h3[CANONICAL_PHI] = 2*M_PI*epsilon;
		h3[CANONICAL_P] =    2*epsilon;
		break;
	}	
}


void VectorCS::randomizeDirection(double m)
{
	(*this) = randomizedDirection(m);
}

VectorCS VectorCS::randomizedDirection(double m) const
{
	VectorCS a = convertedToCoordinateSystem(Cartesian);
	double m1 = a.magnitude();

	a.v[0] += (myrandf() * m);
	a.v[1] += (myrandf() * m);
	a.v[2] += (myrandf() * m);

	a.setMagnitude(m1);
}

static void _scale(double* dest, const double* src, double s, CoordinateSystem cs, int n=3)
{
	if(cs == Cartesian)
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

void VectorCS::scale(double s)
{
	if(cs == Cartesian)
	{
		for(int i=0; i<3; i++)
			v[i] *= s;
	}
	else
	{
		v[0] *= s;
	}
}

VectorCS VectorCS::scaled(double s) const
{
	VectorCS vv = copy();
	vv.scale(s);
	return vv;
}


void VectorCS::scale(double* s3)
{
	for(int i=0; i<3; i++)
		v[i] *= s3[i];
	fix();
}

VectorCS VectorCS::scaled(double* s3) const
{
	VectorCS vv = copy();
	vv.scale(s3);
	return vv;
}



// project a in the b direction
static void _project(double* dest, const double* a, const double* b, CoordinateSystem cs, int n=3)
{
	if(cs == Cartesian)
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
			_convertCoordinateSystem(cs, Cartesian, a, _a);
			_convertCoordinateSystem(cs, Cartesian, b, _b);

			_project(_dest, _a, _b, Cartesian, 3);

			_convertCoordinateSystem(Cartesian, cs, _dest, dest);
		}
		else
		{
			fprintf(stderr, "Don't know how to deal with Spherical or Canonical when dimension != 3\n");
		}
	}
}

// negterm = -1 is standard rejection
// negterm = -2 negates the vector in the given direction
static void _reject(double* dest, const double* a, const double* b, CoordinateSystem cs, int n=3, double neg_term=-1.0)
{
	if(cs == Cartesian)
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
		_convertCoordinateSystem(cs, Cartesian, a, _a);
		_convertCoordinateSystem(cs, Cartesian, b, _b);
		
		_reject(_dest, _a, _b, Cartesian, n, neg_term);
		
		_convertCoordinateSystem(Cartesian, cs, _dest, dest);
	}
}


VectorCS VectorCS::projected(const VectorCS& b) const
{
	double d[3];

	VectorCS bb = b.convertedToCoordinateSystem(cs);
	_project(d, v, bb.v, cs);
	return VectorCS(d,cs);
}

void VectorCS::project(const VectorCS& b)
{
	(*this) = projected(b);
}

VectorCS VectorCS::rejected(const VectorCS& b) const
{
	double d[3];

	VectorCS bb = b.convertedToCoordinateSystem(cs);
	_reject(d, v, bb.v, cs);
	return VectorCS(d,cs);
}

void VectorCS::reject(const VectorCS& b)
{
	(*this) = rejected(b);
}





static double _angleBetween(const double* a, const double* b, CoordinateSystem cs1, CoordinateSystem cs2)
{
	double c1[3];
	double c2[3];
	
	_convertCoordinateSystem(cs1, Cartesian, a, c1);
	_convertCoordinateSystem(cs2, Cartesian, b, c2);

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

double VectorCS::angleBetween(const VectorCS& v1, const VectorCS& v2)
{
	return _angleBetween(v1.v, v2.v, v1.cs, v2.cs);
}


static void _cross(const double* a, const double* b, double* c)
{
	c[0] = a[1]*b[2] - a[2]*b[1];
	c[1] = a[2]*b[0] - a[0]*b[2];
	c[2] = a[0]*b[1] - a[1]*b[0];
}

VectorCS VectorCS::cross(const VectorCS& v1, const VectorCS& v2)
{
	VectorCS c(0,0,0,Cartesian);
	VectorCS a = v1.convertedToCoordinateSystem(Cartesian);
	VectorCS b = v2.convertedToCoordinateSystem(Cartesian);
	_cross(a.v, b.v, c.v);
	c.convertToCoordinateSystem(v1.cs);
	return c;
}


// input vector: a
// requires unit length normal vector: n
// input theta: t
// output vector: b
static void _rotateAboutBy(CoordinateSystem cs, const double* _a, const double* _n, const double t, double* _b)
{
	double a[3];
	double n[3];
	double b[3];

	_convertCoordinateSystem(cs, Cartesian, _a, a);
	_convertCoordinateSystem(cs, Cartesian, _n, n);

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
	
	_convertCoordinateSystem(Cartesian, cs, b, _b);
}


void VectorCS::rotateAboutBy(const VectorCS& n, const double t)
{
	(*this) = rotatedAboutBy(n, t);
}

VectorCS VectorCS::rotatedAboutBy(const VectorCS& n, const double t) const
{
	VectorCS n_cs = n.convertedToCoordinateSystem(cs);
	double b[3];

	_rotateAboutBy(cs, v, n_cs.v, t, b);
	return VectorCS(b, cs);
}


const char* nameOfCoordinateSystem(CoordinateSystem s)
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

CoordinateSystem coordinateSystemByName(const char* name)
{
	if(name == 0)
		return Undefined;
	for(int i=0; i<7; i++)
	{
		const char* c = nameOfCoordinateSystem( (CoordinateSystem) i );
		const int len = strlen(name);

		if(strncasecmp(c, name, len) == 0)
		{
			return (CoordinateSystem) i;
		}
	}
	return Undefined;
}


double VectorCS::arcLength(const VectorCS& v1, const VectorCS& v2)
{
	const double a = VectorCS::angleBetween(v1,v2);
	const double r = v1.magnitude();

	const double c = 2.0 * M_PI * r;
	return c * (a / (2.0 * M_PI));
}

CoordinateSystem baseType(CoordinateSystem cs)
{
	switch(cs)
	{
	case Undefined: return Undefined;
	case Cartesian: return Cartesian;
	case Spherical:
	case SphericalX:
	case SphericalY: return Spherical;
	case Canonical:
	case CanonicalX:
	case CanonicalY: return Canonical;
	}
	return Undefined;
}


bool VectorCS::isType(CoordinateSystem test) const
{
	return baseType(cs) == baseType(test);
}





VectorCS VectorCS::axpy(const double alpha, const VectorCS& x, const VectorCS& _y)
{
	double res[3];

	VectorCS y = _y.convertedToCoordinateSystem(x.cs);
	for(int i=0; i<3; i++)
	{
		res[i] = alpha * x.v[i] + y.v[i];
	}

	return VectorCS(res, x.cs).convertedToCoordinateSystem(x.cs);
}



VectorCS lua_toVectorCS(lua_State* L, int _base, int& consume)
{
    int base = _base;

    if(_base < 0)
	base = lua_gettop(L) + _base + 1;

    CoordinateSystem cs = Cartesian;
    CoordinateSystem _cs = Undefined;
    double x[3] = {0,0,0};	
    int j = 0;
    consume = 0;

    if(lua_istable(L, base))
    {
        for(int i=0; i<4 && j<3; i++)
        {
            lua_pushinteger(L, i+1);
            lua_gettable(L, base);
            if(lua_isnumber(L, -1))
            {
                x[j] = lua_tonumber(L, -1);
                j++;
            }
            lua_pop(L, 1);
        }
        
        for(int i=0; i<4; i++)
        {
            lua_pushinteger(L, i+1);
            lua_gettable(L, base);
            if(_cs == Undefined && lua_isstring(L, -1))
            {
                const char* csn = lua_tostring(L, -1);
                _cs = coordinateSystemByName(csn);
            }
            lua_pop(L, 1);
        }
        if(_cs != Undefined)
            cs = _cs;
        consume = 1;
        return VectorCS(x, cs);
    }
    
    for(int i=0; i<4; i++)
    {
        if(lua_isnumber(L, base+i) && (j < 3))
        {
            x[j] = lua_tonumber(L, base+i);
            j++;
            consume++;
        }
        if((_cs == Undefined) && (lua_isstring(L, base+i)))
        {
            const char* csn = lua_tostring(L, base+i);
            _cs = coordinateSystemByName(csn);
            if(_cs != Undefined)
                consume++;
        }
    }
        
    if(_cs != Undefined)
        cs = _cs;
    
    return VectorCS(x, cs);
}


VectorCS lua_toVectorCS(lua_State* L, int base)
{
	int c;
	return lua_toVectorCS(L, base, c);
}


int lua_pushVectorCS(lua_State* L, const VectorCS& v, const int flags)
{
    if(flags & VCSF_ASTABLE)
    {
        lua_newtable(L);
        for(int i=0; i<3; i++)
        {
            lua_pushinteger(L, i+1);
            lua_pushnumber(L, v.v[i]);
            lua_settable(L, -3);
        }
        
        if(flags & VCSF_CSDESC)
        {
            lua_pushinteger(L, 4);
            lua_pushstring(L, nameOfCoordinateSystem(v.cs));
            lua_settable(L, -3);
        }

        return 1;
    }

    lua_pushnumber(L, v.v[0]);
    lua_pushnumber(L, v.v[1]);
    lua_pushnumber(L, v.v[2]);
    
    if(flags & VCSF_CSDESC)
    {
        lua_pushstring(L, nameOfCoordinateSystem(v.cs));
        return 4;
    }
    return 3;
}

int lua_pushVVectorCS(lua_State* L, std::vector<VectorCS>& vv)
{
    lua_newtable(L);
    for(int i=0; i<vv.size(); i++)
    {
        lua_pushinteger(L, i+1);
        lua_pushVectorCS(L, vv[i], VCSF_ASTABLE | VCSF_CSDESC);
        lua_settable(L, -3);
    }
    return 1;
}


using namespace std;
#include <stdio.h>
void print_vec(const char* msg, const VectorCS& v)
{
	if(msg)
		printf("%s %20s (% 8e, % 8e, % 8e)\n", msg, nameOfCoordinateSystem(v.cs), v.v[0], v.v[1], v.v[2]);
	else
		printf("%20s (% 8e, % 8e, % 8e)\n", nameOfCoordinateSystem(v.cs), v.v[0], v.v[1], v.v[2]);
}
void print_vec(const VectorCS& v)
{
	print_vec((const char*)0, v);
}

void print_vecv(vector<VectorCS>& vv)
{
	for(int i=0; i<vv.size(); i++)
	{
		printf("%3i %20s (% 8e, % 8e, % 8e)\n", i, nameOfCoordinateSystem(vv[i].cs), vv[i].v[0], vv[i].v[1], vv[i].v[2]);
	}
}

