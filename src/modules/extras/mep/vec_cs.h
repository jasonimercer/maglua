#include <luabaseobject.h>

#ifndef VECCS
#define VECCS

// support class and functions for coordinate systems

// Cartesian = x,y,z.
// Spherical = r,phi,theta:  physics convention. (radial, azimuthal, polar) r in R, phi in [0:2pi), theta in [0:pi]
// Canonical = r,phi,p:      r in R, phi in [0:2pi), p in [-1:1] = cos(theta)

enum CoordinateSystem
{
	Undefined =-1,
	Cartesian = 0,
	Spherical = 1,
	Canonical = 2,
	SphericalX = 3,
	SphericalY = 4,
	CanonicalX = 5,
	CanonicalY = 6
};

const char* nameOfCoordinateSystem(CoordinateSystem cs);
CoordinateSystem coordinateSystemByName(const char* name);

class VectorCS
{
public:
    VectorCS() {v[0] = 0; v[1] = 0; v[2] = 0; cs = Cartesian; fix();}
	VectorCS(double v0, double v1, double v2, CoordinateSystem CS=Cartesian)
	{
	    v[0] = v0; v[1] = v1; v[2] = v2; cs = CS; fix();
	}
	VectorCS(const double* vv, CoordinateSystem CS=Cartesian)
	{
	    v[0] = vv[0]; v[1] = vv[1]; v[2] = vv[2]; cs = CS; fix();
	}
	VectorCS(const VectorCS& p)
	{
	    v[0] = p.v[0]; v[1] = p.v[1]; v[2] = p.v[2]; cs = p.cs; fix();
	}
	VectorCS& operator= (const VectorCS& p)
	{
		if(this == &p)
			return *this;
		v[0] = p.v[0]; v[1] = p.v[1]; v[2] = p.v[2]; cs = p.cs;
		return *this;
	}
	VectorCS copy() const
	{
		return VectorCS(v,cs);
	}


	VectorCS normalizedTo(double length=1.0) const;

	void convertToCoordinateSystem(CoordinateSystem new_cs);
	VectorCS convertedToCoordinateSystem(CoordinateSystem new_cs) const;
	bool isType(CoordinateSystem test) const;

	double magnitude() const;
	VectorCS& setMagnitude(double m);

	void scaleFactors(double* sf);
	void stepSize(const double epsilon, double* sf);
	// add a random cartesian vector (scaled by m) to the cartesian 
	// for of the current vector. Rescale to old length, convert back to original CS.
	void randomizeDirection(double m);
	VectorCS randomizedDirection(double m) const;

	void scale(double s);
	VectorCS scaled(double s) const;

	void scale(double* s3);
	VectorCS scaled(double* s3) const;

	void project(const VectorCS& b);
	VectorCS projected(const VectorCS& b) const;

	void reject(const VectorCS& b);
	VectorCS rejected(const VectorCS& b) const;

	void zeroRadialComponent();
	VectorCS zeroedRadialComponent() const;

	void zeroRadialComponent(const VectorCS& radial);
	VectorCS zeroedRadialComponent(const VectorCS& radial) const;


	static double angleBetween(const VectorCS& v1, const VectorCS& v2);
	static double arcLength(const VectorCS& v1, const VectorCS& v2);
	static VectorCS cross(const VectorCS& v1, const VectorCS& v2);
	static VectorCS axpy(const double alpha, const VectorCS& x, const VectorCS& y);

	void rotateAboutBy(const VectorCS& n, const double t);
	VectorCS rotatedAboutBy(const VectorCS& n, const double t) const;
	
	double v[3];
	CoordinateSystem cs;

private:
	void fix(); // fix ranges so they look good 
};


CoordinateSystem baseType(CoordinateSystem cs);

VectorCS lua_toVectorCS(lua_State* L, int base);
VectorCS lua_toVectorCS(lua_State* L, int base, int& consume);

// VectorCS flags
#define VCSF_ASTABLE 0x1 /* use a lua table */
#define VCSF_CSDESC  0x2 /* string form of the coordinate system name appended */
int lua_pushVectorCS(lua_State* L, const VectorCS& v, const int flags=0x0);
int lua_pushVVectorCS(lua_State* L, std::vector<VectorCS>& vv);


// debug
#include <vector>
void print_vec(const char* msg, const VectorCS& v);
void print_vec(const VectorCS& v);
void print_vecv(std::vector<VectorCS>& vv);

#endif

